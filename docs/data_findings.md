# Data findings (2026-09-24, local sample)

Volumes: `ChickenBreastA` (ex vivo, 512 B-scans) and `Fovea1A` (in vivo, 60 B-scans), both
complex64 `(Z, X, Y)`, plus one phantom (`polInt1_polOut1_tomRaw`). `prepare_data` recomputes
these numbers for every volume and stores them in `meta.json` under `diagnostics`.

## 1. Real tomograms carry random bulk phase between A-lines

Complex correlation between A-lines `lag` apart (tissue pixels, top 30 % amplitude):

| volume | | lag 1 | lag 2 | lag 4 | in-band energy K=2 | K=4 |
|---|---|---:|---:|---:|---:|---:|
| ChickenBreastA | raw | 0.23 | 0.07 | 0.01 | 0.62 | 0.34 |
| | bulk-phase corrected | **0.66** | **0.55** | **0.48** | 0.82 | 0.67 |
| Fovea1A | raw | 0.36 | 0.03 | 0.00 | 0.67 | 0.37 |
| | bulk-phase corrected | 0.51 | 0.18 | 0.04 | 0.74 | 0.47 |
| phantom | raw | 0.90 | 0.67 | 0.19 | 1.00 | 0.88 |

Without correction, A-lines two or more apart are uncorrelated in the raw data, so after
decimation the absolute phase of a missing A-line cannot be predicted from the measured ones.
No model can recover it. A network trained on this data can match amplitude but must fail on
phase, which matches the previous experience with this dataset.

The per-A-line phase offset comes from sample motion or trigger jitter. It is removed with the
standard bulk-motion correction of phase-sensitive OCT: the cumulative phase of
Σ_z T(z,x+1)·T*(z,x). This correction also removes any phase common to a whole A-line.
Depth-varying phase, which is what Doppler and OCE measure, survives it
(`tests/test_physics.py::test_bulk_phase_removal_is_independent_of_aline_jitter`).
In an undersampled acquisition the same correction can be estimated from the measured
A-lines, which are K apart; the lag-K correlation above sets how well that works.

## 2. The real data are already near lateral Nyquist

Even after correction, the lateral spectra fill most of the band. The critical factor
K* = 0.5/HW is about 1 for both volumes, versus about 2 for the phantom. Consequences:
- **K=2 is already an aliased, meaningful problem** on real data. It is the default.
- **K=4 is realistic only for the chicken sample** (lag-4 correlation 0.48). For the in-vivo
  fovea it is 0.04: the missing A-lines are close to unpredictable.

## 3. Complex losses shrink amplitude on unmeasured A-lines

With complex + phase losses alone, the model predicted the conditional mean. On unmeasured
A-lines that mean is ~10 dB too dark, which produced vertical stripes and dB-amplitude PSNR
falling below interpolation (19.3 → 14.8). A log-amplitude L1 term (`loss.log_magnitude`,
weight 0.1, now default) fixes it while keeping the complex gains. Local run, K=2, 3000
steps, 2.2M-parameter U-Net:

| val | interpolation | complex + phase | + log-amplitude |
|---|---:|---:|---:|
| PSNR dB-amp ↑ | 19.32 | 14.81 | 18.80 |
| complex NRMSE ↓ | 0.784 | 0.637 | 0.674 |
| \|ρ\| local ↑ | 0.625 | 0.718 | 0.680 |
| weighted phase error [rad] ↓ | 0.444 | 0.421 | 0.423 |

These are smoke-test numbers (two volumes, tiny model), not results.
