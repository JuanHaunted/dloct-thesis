"""
Before/after figure: ground truth vs interpolation (the measurement) vs model, one B-scan per
test sample, with a zoom on the most tissue-rich region.

Rows: ground truth, interpolation, model. Columns: full amplitude (dB) with the zoom box,
zoomed amplitude, zoomed amplitude error vs ground truth, zoomed phase, zoomed inter-A-line
phase difference (phase panels only where signal is >= snr_db above the noise floor).

Writes ``<run>/figures/before_after_<sample>.png`` and ``before_after_<sample>.npz`` (the
cropped complex fields, so the figure can be restyled without the model or the cluster).

    python scripts/figure_before_after.py --run runs/unet_full [--ckpt best] [--zoom 128]
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dloct.data import EvalBScans  # noqa: E402
from dloct.evaluation import amp_dtype, reconstruct  # noqa: E402
from dloct.losses import lateral_phasor  # noqa: E402
from dloct.metrics import compute_metrics, tissue_mask  # noqa: E402
from dloct.models.reconstructors import build_model  # noqa: E402


def load_model(run: Path, ckpt: str, device):
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model = build_model(cfg["model"]).to(device)
    ck = torch.load(run / f"{ckpt}.pt", map_location=device, weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in ck["ema"].items() if k != "n_averaged"})
    return cfg, model.eval(), ck["step"]


def zoom_window(x: torch.Tensor, size: int, snr_db: float):
    """Top-left corner of the size×size window holding the most above-noise signal energy."""
    m = tissue_mask(x[None], snr_db)[0].float() * x.abs() ** 2
    e = F.avg_pool2d(m[None, None], size, stride=8)[0, 0]
    i = int(torch.argmax(e))
    return (i // e.shape[1]) * 8, (i % e.shape[1]) * 8


def to_db(z):
    return 20 * np.log10(np.abs(z) + 1e-12)


def figure(sample, rows, box, snr_db, path, title):
    z0, x0, s = box
    gt = rows["ground truth"]
    mask = tissue_mask(torch.from_numpy(gt)[None], snr_db)[0].numpy()
    sl = (slice(z0, z0 + s), slice(x0, x0 + s))
    mask_c = mask[sl]
    mask_d = mask_c[:, 1:] & mask_c[:, :-1]
    gt_db = to_db(gt)

    cols = ["amplitude [dB]", "zoom: amplitude [dB]", "zoom: amplitude error [dB]",
            "zoom: phase [rad]", "zoom: inter-A-line Δφ [rad]"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.3 * len(cols), 3.1 * len(rows)), squeeze=False)
    for r, (name, z) in enumerate(rows.items()):
        db = to_db(z)
        zc = z[sl]
        dphi = np.angle(lateral_phasor(torch.from_numpy(zc)[None])[0].numpy())
        panels = [
            (db, dict(cmap="gray", vmin=-50, vmax=0)),
            (db[sl], dict(cmap="gray", vmin=-50, vmax=0)),
            (np.clip(db[sl] - gt_db[sl], -15, 15), dict(cmap="RdBu_r", vmin=-15, vmax=15)),
            (np.where(mask_c, np.angle(zc), np.nan), dict(cmap="twilight", vmin=-np.pi, vmax=np.pi)),
            (np.where(mask_d, dphi, np.nan), dict(cmap="twilight", vmin=-np.pi, vmax=np.pi)),
        ]
        for c, (img, kw) in enumerate(panels):
            ax = axes[r, c]
            if name == "ground truth" and c == 2:
                ax.axis("off")
                continue
            im = ax.imshow(img, aspect="auto", interpolation="nearest", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 or (c == 2 and r == 1):
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
                ax.add_patch(Rectangle((x0, z0), s, s, fill=False, ec="yellow", lw=1.2))
            fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best", choices=["best", "latest"])
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", default=None)
    p.add_argument("--zoom", type=int, default=128)
    p.add_argument("--snr-db", type=float, default=10.0)
    args = p.parse_args()

    run = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, step = load_model(run, args.ckpt, device)
    K = cfg["data"]["factor"]
    divisor = math.lcm(model.divisor, K)
    ds = EvalBScans(args.data_root or cfg["data"]["root"], args.split, per_volume=None,
                    divisor=divisor, sources=cfg["data"].get("sources"))

    # One B-scan per sample: the middle B-scan of its first volume.
    chosen = {}
    for i, (name, y) in enumerate(ds.items):
        sample = ds.vols.volume(name)["group"].split("/", 1)[-1]
        chosen.setdefault(sample, []).append(i)
    out = run / "figures"
    out.mkdir(exist_ok=True)
    dtype = amp_dtype(cfg["train"].get("precision", "auto"))
    for sample, idx in sorted(chosen.items()):
        i = idx[len(idx) // 2]
        x, name, y = ds[i]
        x = x.to(device)[None]
        recon = reconstruct(model, x, K, amp_dtype=dtype)
        rows = {"ground truth": x[0].cpu().numpy(),
                "interpolation (before)": recon["interpolation"][0].cpu().numpy(),
                "model (after)": recon["model"][0].cpu().numpy()}
        z0, x0 = zoom_window(x[0].cpu(), args.zoom, args.snr_db)
        m_i = compute_metrics(recon["interpolation"], x, args.snr_db)
        m_m = compute_metrics(recon["model"], x, args.snr_db)
        title = (f"{cfg['name']} (step {step}), {sample} {name.split('__')[-1]} y={y}, K={K}   |   "
                 f"PSNR {m_i['psnr_db']:.2f} → {m_m['psnr_db']:.2f} dB, "
                 f"WPC {m_i['wpc']:.3f} → {m_m['wpc']:.3f}, CCC {m_i['ccc']:.3f} → {m_m['ccc']:.3f}")
        figure(sample, rows, (z0, x0, args.zoom), args.snr_db, out / f"before_after_{sample}.png", title)
        np.savez_compressed(out / f"before_after_{sample}.npz", volume=name, y=y, factor=K, zoom=[z0, x0, args.zoom],
                            **{k.split(" ")[0]: v for k, v in rows.items()})
        print(f"{sample}: {name} y={y} zoom=({z0},{x0}) {title.split('|')[1].strip()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
