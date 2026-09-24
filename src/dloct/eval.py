"""
Evaluate a trained run on the test split (full B-scans) and write thesis-ready outputs.

    python -m dloct.eval --run runs/unet_full [--ckpt best] [--split test] [--per-volume 32]

Writes to ``<run>/eval_<split>/``: ``metrics.json``, ``metrics.md`` (one table per data
source), ``mps.png`` (lateral spectra: is the fold undone?) and ``example_*.png`` comparison
figures. Uses the EMA weights.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from .data import EvalBScans
from .evaluation import evaluate
from .models.reconstructors import build_model
from .sampling_analysis import compute_spectral_halfwidth
from .visualize import comparison_figure, mps_figure

COLUMNS = [
    ("psnr_db", "PSNR dB-amp ↑"), ("ssim_db", "SSIM dB-amp ↑"), ("nrmse", "cNRMSE ↓"),
    ("rho_global", "\\|ρ\\| ↑"), ("rho_local", "\\|ρ\\| local ↑"), ("phase_err_rad", "φ err [rad] ↓"),
    ("dphase_err_rad", "Δφ err [rad] ↓"), ("speckle_contrast", "speckle C"),
]


def markdown_table(summary: dict) -> str:
    head = "| method | " + " | ".join(c for _, c in COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(COLUMNS)
    rows = [f"| {m} | " + " | ".join(f"{v[k]:.4f}" for k, _ in COLUMNS) + " |" for m, v in summary.items()]
    return "\n".join([head, sep, *rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best", choices=["best", "latest"])
    p.add_argument("--split", default="test")
    p.add_argument("--per-volume", type=int, default=32, help="B-scans per volume (0 = all)")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--data-root", default=None, help="override data.root from the run config")
    args = p.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    dcfg = cfg["data"]
    K = dcfg["factor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    ck = torch.load(run / f"{args.ckpt}.pt", map_location=device, weights_only=False)
    ema = {k.removeprefix("module."): v for k, v in ck["ema"].items() if k != "n_averaged"}
    model.load_state_dict(ema)
    divisor = math.lcm(model.divisor, K)

    ds = EvalBScans(args.data_root or dcfg["root"], args.split, per_volume=args.per_volume or None,
                    divisor=divisor)
    summary, per_source, examples, spectra = evaluate(model, ds, K, device, dcfg["tissue_db"],
                                                      bf16=cfg["train"]["bf16"], keep=args.examples)

    out = run / f"eval_{args.split}"
    out.mkdir(exist_ok=True)
    halfwidth = {m: compute_spectral_halfwidth(f, s / s.max(), 0.01).half_width for m, (f, s) in spectra.items()}
    (out / "metrics.json").write_text(json.dumps(dict(
        step=ck["step"], ckpt=args.ckpt, split=args.split, n_bscans=len(ds), factor=K,
        summary=summary, per_source=per_source, mps_halfwidth_1pct=halfwidth), indent=2))

    md = [f"# {cfg['name']} — {args.split} (K={K}, {len(ds)} B-scans, step {ck['step']}, {args.ckpt})",
          "", "## All", "", markdown_table(summary)]
    for source, s in per_source.items():
        md += ["", f"## {source}", "", markdown_table(s)]
    md += ["", "## Lateral MPS half-width at 1% of peak", "",
           *[f"- {m}: {hw:.4f}" for m, hw in halfwidth.items()]]
    (out / "metrics.md").write_text("\n".join(md) + "\n")

    mps_figure(spectra, out / "mps.png", K)
    for i, (name, gt, preds) in enumerate(examples):
        comparison_figure(gt, preds, out / f"example_{i}.png", dcfg["tissue_db"], title=f"{cfg['name']} — {name}")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
