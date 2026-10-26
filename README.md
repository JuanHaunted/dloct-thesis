# DLOCT

Deep learning reconstruction of laterally undersampled complex OCT tomograms. B-scans are
undersampled along the fast axis (every K-th A-line is kept) and a network recovers both the
amplitude and the phase of the full complex field.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Data

Put the raw tomograms (`.npy`, complex or with a trailing Re/Im axis, layout Z, X, Y) in
`data/train/phase/`, then convert them once:

```bash
uv run python scripts/check_data.py data/train/phase
uv run python -m dloct.prepare_data --src data/train --out data/prepared --split-file configs/split.yaml
```

This reorders the volumes by B-scan, corrects the bulk phase between A-lines, normalizes each
volume and assigns the train/validation/test split by sample. See `data/README.md` for details.

## Training

Each experiment is a YAML file in `configs/` that inherits from `configs/base.yaml`. Any setting
can be overridden from the command line.

```bash
uv run python -m dloct.train --config configs/unet_full.yaml
uv run torchrun --standalone --nproc_per_node=2 -m dloct.train --config configs/unet_full.yaml
uv run python -m dloct.train --config configs/unet_full.yaml --set data.factor=4 name=unet_k4
```

Outputs go to `runs/<name>/`. A run resumes from `runs/<name>/latest.pt` if it exists.

| Config | Description |
|---|---|
| `unet_full` | Single-shot U-Net with the phase-aware loss |
| `cascade_full` | Unrolled cascade of U-Nets with data consistency |
| `unet_magnitude`, `unet_complex` | Loss ablations (amplitude only, no phase terms) |
| `unet_ft`, `unet_power`, `unet_gan`, `unet_gan_power`, `unet_gan_power_lo` | Fine-tunes of `unet_full` (control, power matching, discriminator, combinations) |
| `cascade_ft`, `cascade_gan` | Fine-tunes of `cascade_full` |
| `unet_k4`, `cascade_k4` | Undersampling factor K = 4 |

Fine-tune configs start from the parent run's `best.pt`, so the parent has to be trained first.

## Evaluation

```bash
uv run python -m dloct.eval --run runs/unet_full --ckpt best
uv run python -m dloct.compare --ref runs/unet_full/eval_test_best_snr10 --cand runs/unet_power/eval_test_latest_snr10
```

`eval` writes metrics with confidence intervals and example figures to
`runs/<name>/eval_test_<ckpt>_snr10/`. `compare` runs paired tests between two evaluated runs.

Figures comparing all models:

```bash
uv run python scripts/dump_reconstructions.py
uv run python scripts/make_figures.py
uv run python scripts/make_extra_figures.py
```

## Cluster

SLURM scripts for the EAFIT Apolo cluster are in `scripts/` (`train.slurm`, `eval.slurm`,
`prepare.slurm`). Setup and usage are described in `docs/apolo.md`.

```bash
sbatch scripts/train.slurm configs/unet_full.yaml
sbatch scripts/eval.slurm runs/unet_full --ckpt best
```

## Documentation

- `docs/status.md`: experiments, results and decisions
- `docs/data_findings.md`: properties of the dataset
- `docs/literature_review.md`: related work
