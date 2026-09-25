# Project status

_Maintained by the programming agent. Last updated 2026-09-25._

## Problem and approach

Reconstruct laterally undersampled **complex** OCT B-scans (amplitude and phase). Measurement:
keep every K-th A-line (K=2, no anti-alias filter), sinc-interpolate back to the full grid
(`src/dloct/physics.py`). Networks see Re/Im plus a mask of the measured A-lines.

- **Model A: `unet_full`.** Single-shot ConvNeXt U-Net (8.7M parameters), residual on the
  interpolated measurement. Data consistency (re-inserting the measured A-lines) is applied only at inference.
- **Model B: `cascade_full`.** Unrolled physics-informed cascade: 5 × (small U-Net → exact data
  consistency), trained end to end through the DC layers.
- **Ablations:** `unet_magnitude` (amplitude-only L1, as prior work does) and `unet_complex`
  (no explicit phase terms).
- **Loss:** complex Charbonnier + 0.1 log-amplitude L1 + 0.1 amplitude-weighted (1−cos Δφ) +
  0.1 inter-A-line phase-difference term + 0.01 lateral-spectrum L1.

## Key decisions (with reasons)

| Decision | Reason |
|---|---|
| No cold diffusion | Earlier attempts corrupted signals. Lit review: modest gains over DC cascades, and stochastic diffusion hallucinates speckle phase |
| Real tomograms only | User's choice. Phantoms are not representative |
| Bulk-phase correction in preprocessing | Raw A-lines have random phase offsets (motion/trigger jitter): lag-2 complex correlation ≈ 0.03–0.07 raw, 0.12–0.55 corrected. Without it, missing-A-line phase is unpredictable |
| K=2 (not 4) | Data are at lateral Nyquist (K* ≈ 1). After correction, lag-4 correlation is only 0.02–0.18 on in-vivo samples |
| Linear per-volume normalization (P99.9) | The forward operator is linear, and log compression would break it |
| Log-amplitude loss term | Without it, the unmeasured A-lines came out ~10 dB too dark (conditional-mean shrinkage) |
| Per-sample split, A/B channels together | Prevents leakage between the two channels of one acquisition |
| Per-sample balanced patch sampling | Four ~1000-B-scan volumes would otherwise dominate training |
| Phase metrics at ≥ 10 dB above noise floor | A fixed −30 dB mask counted 88% of fovea pixels (mostly noise) as tissue |

## Data

29 files = 15 samples (A/B channels grouped), from two systems: retina/optic nerve on the
ophthalmic SS-OCT, and the other tissues on the benchtop system.

| Split | Samples |
|---|---|
| train | ChickenBreast, Fovea1–3, OpticNerve1–2, OpticNerveNew, OpticNerveOld, S.Eye2, unpairCadaverhearth |
| val | Fovea4, OpticNerve3 |
| test | Fovea5, OpticNerve4, Nail (Nail = tissue type unseen in training) |

Per-volume coherence diagnostics are in `meta.json` on the cluster and summarized in
`docs/data_findings.md` (local sample only).

## Results

### `unet_full`: validation (Fovea4, OpticNerve3), 100k steps, best checkpoint at step 32.5k

Validation metrics use the **old** −30 dB mask, so the phase numbers are understated.
Test-set numbers with the corrected mask are pending.

| | interpolation | unet_full |
|---|---:|---:|
| PSNR dB-amplitude | 19.30 | 21.62 |
| SSIM dB-amplitude | 0.519 | 0.656 |
| complex NRMSE | 0.766 | 0.644 |
| \|ρ\| local | 0.628 | 0.688 |
| phase error, amplitude-weighted [rad] | 0.473 | 0.437 |
| inter-A-line Δφ error [rad] | 0.618 | 0.592 |

Observations: large amplitude gain and modest phase gain. Phase metrics overfit after ~35k steps
while amplitude keeps improving. Data consistency adds ~nothing, because the model already
preserves the measured A-lines.

### Pending

- [ ] `unet_full` test evaluation with SNR mask, WPC/CCC/PG-SSIM and CIs (best and latest checkpoints)
- [ ] `cascade_full` training (about 19 h total), then evaluation
- [ ] `unet_magnitude`, `unet_complex` training and evaluation

## Open questions

- Are any differently named samples the same physical tissue (e.g. OpticNerve4 vs OpticNerveNew)?
  The user confirmed the split as is.
- Optional second iteration: phase-loss weight 0.2, and earlier stopping given phase overfitting.
