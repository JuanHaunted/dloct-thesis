# Project status

_Maintained by the programming agent. Last updated 2026-09-25 (unet_full test results)._

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

### `unet_full`: test set (Fovea5, OpticNerve4, Nail), final metrics

Source: `runs/unet_full/eval_test_best_snr10/metrics.md` (best checkpoint, step 32.5k) and
`eval_test_latest_snr10/` (step 100k). 192 B-scans (64 per sample). Values are means over
B-scans with [95% bootstrap CI]. Phase metrics use pixels ≥ 10 dB above the noise floor (17.8% of pixels).

| metric | interpolation | unet_full (best) | paired Δ [95% CI] | B-scans improved |
|---|---:|---:|---:|---:|
| PSNR dB-amplitude ↑ | 19.54 [19.49, 19.59] | **21.97** [21.89, 22.05] | +2.44 [+2.40, +2.47] dB | 100% |
| SSIM dB-amplitude ↑ | 0.519 | **0.660** | +0.141 (+27%) | 100% |
| complex NRMSE ↓ | 0.751 | **0.638** | −0.114 (−15%) | 100% |
| \|ρ\| local ↑ | 0.719 | **0.764** | +0.045 | 100% |
| WPC ↑ | 0.791 [0.784, 0.797] | **0.834** [0.828, 0.840] | +0.043 | 100% |
| CCC ↑ | 0.745 [0.739, 0.751] | **0.794** [0.788, 0.799] | +0.049 | 100% |
| PG-SSIM ↑ | 0.203 | **0.217** | +0.013 | 97.9% |
| phase error, amplitude-weighted [rad] ↓ | 0.385 | **0.345** | −0.040 (−10%) | 99.5% |
| inter-A-line Δφ error [rad] ↓ | 0.389 | **0.364** | −0.025 (−6%) | 96.9% |

Per sample (best checkpoint), all three improve on every metric:

| sample | PSNR interp → model | WPC interp → model | CCC interp → model | Δφ err interp → model |
|---|---|---|---|---|
| Fovea5 | 19.32 → 21.65 | 0.796 → 0.841 | 0.767 → 0.815 | 0.290 → 0.259 |
| OpticNerve4 | 19.29 → 21.56 | 0.764 → 0.812 | 0.736 → 0.785 | 0.398 → 0.376 |
| Nail (unseen tissue) | 20.00 → 22.71 | 0.813 → 0.849 | 0.732 → 0.782 | 0.477 → 0.457 |

Best vs final checkpoint: step 100k has slightly better amplitude (PSNR 22.08, SSIM 0.664) and
slightly worse phase (WPC 0.830, Δφ error 0.378). This is the amplitude–phase trade-off over
training, and the best checkpoint (selected on validation |ρ| local) is the phase-favouring one.

Lateral spectrum (`mps.png`): interpolation has excess energy near the band edge (aliased energy
folded back) and none beyond it. The model matches the ground truth in-band (the fold is undone)
and restores part of the out-of-band energy, but ends ~8 dB below the ground truth at the Nyquist
edge (−16 vs −8 dB): the finest lateral scales are still under-recovered. An out-of-band recovery
metric was added to `eval.py` after this evaluation, so re-evaluation reports it as a number.

Mask effect: with the old fixed −30 dB mask the phase-error reduction was −8.1% (weighted) and
−4.5% (Δφ); with the SNR mask it is −10.3% and −6.3%. The old mask understated the phase gains.

Data consistency: no measurable effect. The model already reproduces the measured A-lines.

### `unet_full`: validation, for reference

Best at step 32.5k. Phase metrics overfit after ~35k steps while amplitude keeps improving.

### Pending

- [ ] `cascade_full` training (about 19 h total), then evaluation
- [ ] `unet_magnitude`, `unet_complex` training and evaluation
- [ ] Re-run `unet_full` evaluation to get the out-of-band spectral recovery numbers (optional)
- [ ] Before/after figure for `unet_full` (`scripts/figure_before_after.py`, running on Apolo)

### Next architecture (decided 2026-09-25)

Adversarial training (a discriminator for realistic speckle) on top of **both** `unet_full` and
`cascade_full`, compared against those same networks without it. It will be implemented after the
`unet_full` before/after figure is reviewed, and launched after the cascade results.
`unet_phase2` (phase terms ×2, 40k steps) is on hold. A K=4 run is possible next week.

## Open questions

- Are any differently named samples the same physical tissue (e.g. OpticNerve4 vs OpticNerveNew)?
  The user confirmed the split as is.
- Adversarial models: which metrics show the benefit? Discriminators usually improve realism
  (speckle statistics, high-frequency spectrum) and can cost pixel-wise distortion (PSNR, WPC):
  the perception–distortion trade-off, Blau & Michaeli 2018. Plan to report both kinds.
