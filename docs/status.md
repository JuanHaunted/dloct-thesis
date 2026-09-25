# Project status

_Maintained by the programming agent. Last updated 2026-09-25 (final unet_full test evaluation)._

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

## Statistical methodology (from 2026-09-25)

- Per-B-scan metrics, paired across methods by (volume, B-scan index).
- **Two 95% bootstrap CIs:** over B-scans (optimistic, since neighbouring B-scans are correlated)
  and over volumes (per-volume means resampled; honest but wide with 6 test volumes). A/B channels of
  one sample are still correlated with each other, so even the volume CI is somewhat optimistic.
- **Significance:** paired two-sided Wilcoxon signed-rank tests, Holm–Bonferroni-corrected across
  the tested metrics. They are run over B-scan pairs and over per-volume means. The volume-level
  test is conservative: with 6 volumes the smallest possible p is 0.031, so after Holm correction it
  cannot reach 0.05. Report it as descriptive.
- **Pre-registered decision rule** for "model B is better than model A" (`dloct.compare`): B
  improves the main metric (tissue PSNR), no phase metric (WPC, CCC, PG-SSIM) degrades by more than
  5%, and WPC or CCC improves with Holm-adjusted p < 0.05.
- **Extra diagnostics:** HistSim (amplitude histogram similarity), phase coherence by amplitude
  decile, SSIM between output and input (identity-collapse check).
- The CIs quoted in the `unet_full` test table below are the older B-scan-level ones; re-evaluation
  will add volume-level CIs and the tests.

## Data

**Duplicate volume (found 2026-09-25):** `Fovea5A.npy` and `Fovea5B.npy` give byte-for-byte
identical metrics on every evaluated B-scan, so they hold the same data. The group's own
evaluation code also excludes a duplicated volume. Consequences:
- No leakage: both are in the test split, same sample.
- The test set has **5 distinct volumes**, not 6. Numbers below with "5 volumes" exclude Fovea5B.
- Training may contain duplicates too. They don't bias sampling (balancing is per sample), but
  the training set is smaller than it looks. `scripts/check_data.py` now lists byte-identical
  files, and `prepare_data` excludes them from all splits. This needs checking on Apolo.

29 files = 15 samples (A/B channels grouped), from two systems: retina/optic nerve on the
ophthalmic SS-OCT, and the other tissues on the benchtop system.

| Split | Samples |
|---|---|
| train | ChickenBreast, Fovea1–3, OpticNerve1–2, OpticNerveNew, OpticNerveOld, S.Eye2, unpairCadaverhearth (10 samples, 19 volumes) |
| val | Fovea4, OpticNerve3 |
| test | Fovea5, OpticNerve4, Nail (Nail = tissue type unseen in training); 5 volumes (Fovea5B is a duplicate, excluded) |

Per-volume coherence diagnostics are in `meta.json` on the cluster and summarized in
`docs/data_findings.md` (local sample only).

## Results

### `unet_full`: final test evaluation (authoritative)

Source: `runs/unet_full/eval_test_best_snr10/metrics.md` (best checkpoint, step 32.5k; duplicates
excluded; 5 volumes, 160 B-scans, 32 per volume). Brackets are **volume-level** 95% bootstrap CIs.
Paired Wilcoxon over B-scans, Holm-corrected across 11 metrics: p ≤ 5.5e-21 for every metric.
The volume-level test gives p = 0.69; with 5 volumes it cannot go below 0.0625, so it is descriptive.

| metric | interpolation | unet_full | Δ [95% CI by volume] | B-scans improved |
|---|---:|---:|---:|---:|
| **Phase** | | | | |
| WPC ↑ | 0.790 [0.769, 0.810] | 0.833 [0.817, 0.848] | +0.043 [+0.038, +0.048] | 100% |
| CCC ↑ | 0.740 [0.726, 0.755] | 0.790 [0.776, 0.803] | +0.049 [+0.048, +0.051] | 100% |
| PG-SSIM ↑ | 0.200 | 0.212 | +0.013 [+0.011, +0.015] | 98% |
| phase error, amplitude-weighted [rad] ↓ | 0.394 | 0.356 | −0.038 [−0.044, −0.033] | 99% |
| inter-A-line Δφ error [rad] ↓ | 0.408 | 0.385 | −0.023 [−0.028, −0.019] | 96% |
| **Complex field** | | | | |
| complex NRMSE ↓ | 0.754 | 0.641 | −0.113 [−0.121, −0.105] | 100% |
| \|ρ\| local ↑ | 0.707 | 0.752 | +0.045 [+0.044, +0.047] | 100% |
| **Amplitude, tissue only** | | | | |
| PSNR tissue ↑ | 20.13 [19.90, 20.36] | 21.19 [20.05, 22.33] | +1.06 [+0.10, +2.03] | 87% |
| SSIM tissue ↑ | 0.572 | 0.670 | +0.098 [+0.083, +0.114] | 100% |
| **Realism / artifacts** | | | | |
| HistSim (amplitude histogram) ↑ | 0.9996 | 0.854 | −0.146 [−0.185, −0.117] | 0% (worse) |
| unmeasured A-line bias [dB] (→ 0) | −3.22 | −4.71 | \|bias\| +1.50 (worse) | 0% |
| unmeasured/measured power (GT 1.007) | 0.790 | 0.425 | | |
| SSIM output vs input (identity check) | 1.000 | 0.699 | not collapsed to identity | |
| **Spectrum** | | | | |
| out-of-band energy recovered | 0.000 | 0.270 | | |
| out-of-band / in-band spectral error [dB] | 52.7 / 1.60 | 6.4 / 0.75 | | |

Per sample, tissue PSNR interpolation → model: Fovea5 20.44 → 20.88 (+0.4), Nail 20.26 → 22.69
(+2.4), OpticNerve4 19.84 → 19.85 (+0.0). Phase metrics improve on all three (e.g. WPC 0.796 → 0.841,
0.813 → 0.849, 0.764 → 0.812).

Phase coherence by amplitude decile: the gain grows with signal strength, from +0.003 in d1
(noise) to +0.025 in d10 (brightest tissue). Phase is recovered where the signal supports it.

**Interpretation:** a consistent, significant phase and complex-field gain on every sample. The
amplitude gain in tissue is sample-dependent: large for Nail, ~0 for the retina. Realism gets worse:
the unmeasured A-lines are darker than interpolation's (striping) and the background noise is
smoothed (HistSim 0.85). This is the conditional-mean behaviour that the adversarial step targets.

### `unet_full`: test set, original table (6 volumes, Fovea5 counted twice; superseded)

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

**Caveat, found from the before/after figures (2026-09-25): the whole-image PSNR/SSIM above
are dominated by background noise.** Only 11–40% of pixels are tissue. The model smooths the
background noise (dB spread 5.8 → ~3 dB), which whole-image PSNR rewards. Split by region, from
`runs/unet_full/figures/before_after_*.npz`, one B-scan per sample:

| PSNR dB-amp | whole image | tissue | background |
|---|---|---|---|
| Fovea5 | 19.36 → 21.67 | 20.26 → 20.70 | 19.24 → 21.85 |
| OpticNerve4 | 19.22 → 21.52 | 19.95 → 19.53 | 19.13 → 21.84 |
| Nail | 19.85 → 22.32 | 20.06 → 22.47 | 19.72 → 22.22 |

So for retina the amplitude gain in tissue is ~0 dB; Nail gains in tissue too. **Do not quote
the whole-image PSNR/SSIM as an amplitude improvement.** The phase metrics are computed in
tissue and are unaffected.

**Striping artifact:** in tissue, the model's unmeasured A-lines are 5–6 dB too dark (power
ratio unmeasured/measured 0.32–0.57, vs ground truth ≈ 1.0 and interpolation 0.74–0.95). This is
conditional-mean shrinkage: where the missing A-line is only partly predictable, the MSE-optimal
estimate has a smaller magnitude. It is visible as vertical striping in the zoomed amplitude.
Interpolation shows the same effect more weakly (−3.6 to −3.9 dB). This is the main motivation
for the adversarial step.

`eval.py` now reports tissue-only PSNR/SSIM (`psnr_db_tissue`, `ssim_db_tissue`), the dB bias on
unmeasured A-lines (`unmeasured_db_bias`) and their power ratio (`unmeasured_power_ratio`, plus
`_gt`). `unet_full` should be re-evaluated to get them on the full test set.

Per sample (best checkpoint), all three improve on every phase metric:

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
- [x] Before/after figure for `unet_full`: `runs/unet_full/figures/` (revealed the caveat above)
- [ ] Re-evaluate `unet_full` with the tissue and unmeasured-line metrics

### Adversarial step (implemented 2026-09-25, smoke-tested locally)

All runs fine-tune a trained model from its EMA weights for **20k steps** (lr 5e-5, same losses),
so the comparison is "same network, same start, one change". Final (`latest`) checkpoints are
evaluated for all fine-tunes, to avoid checkpoint-selection bias.

| Run | Starts from | Change | Answers |
|---|---|---|---|
| `unet_ft` | unet_full | none (control) | is any gain just more training? |
| `unet_gan` | unet_full | conditional PatchGAN on dB amplitude, hinge loss, weight 0.01 | does a speckle-realism discriminator help? |
| `unet_power` | unet_full | power-match loss on unmeasured A-lines (0.1) | is it just brightness correction? |
| `cascade_ft` | cascade_full | none (control) | as above, for the cascade |
| `cascade_gan` | cascade_full | same discriminator | as above, for the cascade |

- **Discriminator** (`src/dloct/models/discriminator.py`): 3-layer PatchGAN with spectral norm
  (2.8M parameters). It is conditioned on the measurement and the measured-A-line mask. Input is
  dB amplitude only, so phase stays under the physics-aware losses; a `logphasor` mode also exists.
- **Hypotheses**, from the realism diagnostics of unet_full: GAN > ft on `hist_sim`,
  `unmeasured_power_ratio` (→ ~1.0), `unmeasured_db_bias` (→ 0) and out-of-band spectral
  recovery, with no phase metric degrading by more than 5%.
- **Decision rule**, main metric tissue PSNR: `python -m dloct.compare --ref <ft eval> --cand <gan eval>`.
- The adversarial weight 0.01 is a first guess. Refine after the cascade and the first GAN run.

## Open questions

- Are any differently named samples the same physical tissue (e.g. OpticNerve4 vs OpticNerveNew)?
  The user confirmed the split as is.
- Adversarial models: which metrics show the benefit? Discriminators usually improve realism
  (speckle statistics, high-frequency spectrum) and can cost pixel-wise distortion (PSNR, WPC):
  the perception–distortion trade-off, Blau & Michaeli 2018. Plan to report both kinds.
