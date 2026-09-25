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

import numpy as np
import torch
import yaml

from .data import EvalBScans
from .evaluation import amp_dtype, bootstrap_ci, evaluate
from .models.reconstructors import build_model
from .sampling_analysis import compute_spectral_halfwidth
from .visualize import comparison_figure, mps_figure

AMPLITUDE_COLUMNS = [
    ("psnr_db_tissue", "PSNR tissue ↑"), ("ssim_db_tissue", "SSIM tissue ↑"),
    ("psnr_db", "PSNR image ↑"), ("ssim_db", "SSIM image ↑"),
    ("unmeasured_db_bias", "unmeasured bias [dB] →0"), ("unmeasured_power_ratio", "unmeasured/measured power"),
]
COMPLEX_COLUMNS = [
    ("nrmse", "cNRMSE ↓"), ("rho_global", "\\|ρ\\| ↑"), ("rho_local", "\\|ρ\\| local ↑"),
]
PHASE_COLUMNS = [
    ("wpc", "WPC ↑"), ("ccc", "CCC ↑"), ("pg_ssim", "PG-SSIM ↑"),
    ("phase_err_w_rad", "φ err [rad] ↓"), ("dphase_err_rad", "Δφ err [rad] ↓"),
]


def markdown_table(summary: dict, columns, ci: dict | None = None) -> str:
    """One row per method; with ``ci`` each cell reads ``mean [lo, hi]``."""
    head = "| method | " + " | ".join(c for _, c in columns) + " |"
    sep = "|---|" + "---:|" * len(columns)
    def cell(method, k):
        s = f"{summary[method][k]:.4f}"
        if ci:
            lo, hi = ci[method][k]
            s += f" [{lo:.4f}, {hi:.4f}]"
        return s
    rows = [f"| {m} | " + " | ".join(cell(m, k) for k, _ in columns) + " |" for m in summary]
    return "\n".join([head, sep, *rows])


def spectral_recovery(spectra: dict, factor: int) -> dict:
    """
    Per method, compared with the ground-truth lateral MPS: the fraction of out-of-band energy
    recovered, and the mean absolute dB error of the spectrum out of and inside the band.
    Spectra are in dB relative to the ground-truth peak and clipped at −60 dB, so a band with
    no energy at all scores its distance to that floor rather than an arbitrary number.
    """
    f, gt = spectra["ground truth"]
    oob = np.abs(f) > 0.5 / factor   # the band-edge bin holds half the measured Nyquist bin
    ref = gt.max()
    to_db = lambda p: np.maximum(10 * np.log10(p / ref + 1e-30), -60.0)
    gt_db = to_db(gt)
    out = {}
    for method, (_, s) in spectra.items():
        if method == "ground truth":
            continue
        s_db = to_db(s)
        out[method] = dict(
            oob_energy_ratio=float(s[oob].sum() / gt[oob].sum()),
            oob_db_error=float(np.abs(s_db - gt_db)[oob].mean()),
            inband_db_error=float(np.abs(s_db - gt_db)[~oob].mean()),
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best", choices=["best", "latest"])
    p.add_argument("--split", default="test")
    p.add_argument("--per-volume", type=int, default=32, help="B-scans per volume (0 = all)")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--data-root", default=None, help="override data.root from the run config")
    p.add_argument("--snr-db", type=float, default=10.0,
                   help="phase metrics only where the signal is this far above the noise floor")
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
                    divisor=divisor, sources=dcfg.get("sources"))
    summary, per_sample, examples, spectra, records = evaluate(
        model, ds, K, device, args.snr_db, amp_dtype=amp_dtype(cfg["train"].get("precision", "auto")),
        keep=args.examples, return_records=True)
    ci = {method: {k: bootstrap_ci([m[k] for _, mt, m in records if mt == method]) for k in summary[method]}
          for method in summary}

    out = run / f"eval_{args.split}_{args.ckpt}_snr{args.snr_db:g}"
    out.mkdir(exist_ok=True)
    halfwidth = {m: compute_spectral_halfwidth(f, s / s.max(), 0.01).half_width for m, (f, s) in spectra.items()}
    spectral = spectral_recovery(spectra, K)
    (out / "metrics.json").write_text(json.dumps(dict(
        step=ck["step"], ckpt=args.ckpt, split=args.split, n_bscans=len(ds), factor=K,
        summary=summary, ci95=ci, per_sample=per_sample, mps_halfwidth_1pct=halfwidth, spectral=spectral,
        per_bscan=[dict(sample=s, method=mt, **m) for s, mt, m in records]), indent=2))

    md = [f"# {cfg['name']} — {args.split} (K={K}, {len(ds)} B-scans, step {ck['step']}, {args.ckpt}, "
          f"phase metrics at SNR >= {args.snr_db:g} dB, {summary['interpolation']['mask_fraction']:.0%} of pixels)",
          "", "Mean over B-scans [95% bootstrap CI].", "",
          "Amplitude in dB. 'tissue' = pixels ≥ SNR threshold above the noise floor; 'image' = all",
          "pixels, dominated by background noise. Ground-truth unmeasured/measured power ≈ "
          f"{summary['interpolation'].get('unmeasured_power_ratio_gt', float('nan')):.3f}.", "",
          "## All: amplitude", "", markdown_table(summary, AMPLITUDE_COLUMNS, ci),
          "", "## All: complex field", "", markdown_table(summary, COMPLEX_COLUMNS, ci),
          "", "## All: phase", "", markdown_table(summary, PHASE_COLUMNS, ci)]
    for sample, s in sorted(per_sample.items()):
        md += ["", f"## {sample}", "", markdown_table(s, AMPLITUDE_COLUMNS), "",
               markdown_table(s, COMPLEX_COLUMNS), "", markdown_table(s, PHASE_COLUMNS)]
    md += ["", "## Lateral spectrum", "",
           "Out-of-band = |f| > 1/(2K), the band the measurement cannot contain. Recovered energy is",
           "relative to the ground truth (1 = fully restored). Spectral error is the mean |ΔdB| of",
           "the MPS against the ground truth in each band (spectra clipped at −60 dB).", "",
           "| method | out-of-band energy recovered | out-of-band spectral error [dB] | in-band spectral error [dB] | half-width @1% |",
           "|---|---:|---:|---:|---:|",
           *[f"| {m} | {s['oob_energy_ratio']:.3f} | {s['oob_db_error']:.2f} | {s['inband_db_error']:.2f} | {halfwidth[m]:.4f} |"
             for m, s in spectral.items()]]
    (out / "metrics.md").write_text("\n".join(md) + "\n")

    mps_figure(spectra, out / "mps.png", K)
    for i, (name, gt, preds) in enumerate(examples):
        comparison_figure(gt, preds, out / f"example_{i}.png", args.snr_db, title=f"{cfg['name']} — {name}")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
