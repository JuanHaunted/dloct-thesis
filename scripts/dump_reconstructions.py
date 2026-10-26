"""
Reconstruct one test B-scan per sample with every model, for qualitative comparison figures.

Writes ``runs/figures/recon_<sample>.npz`` with the ground truth, the interpolated measurement
and each model's output (complex64, full B-scan), plus metadata. ``scripts/make_figures.py``
renders the figures from these files, locally, without the models.

    python scripts/dump_reconstructions.py [--data-root ~/dloct/prepared]

By default the B-scan is the middle one of the first test volume of each sample (the same choice
as ``figure_before_after.py``). ``--volume NAME --y INDEX`` reconstructs one specific test B-scan
instead (e.g. ``--volume phase__NailB --y 165``), written to ``recon_<sample>_<volume>_y<INDEX>.npz``.
Models whose checkpoint is missing are skipped.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dloct.data import EvalBScans  # noqa: E402
from dloct.evaluation import amp_dtype  # noqa: E402
from dloct.models.reconstructors import build_model  # noqa: E402
from dloct.physics import measure  # noqa: E402

# (label, run, checkpoint). Keep labels in sync with MODELS in make_figures.py.
MODELS = [
    ("U-Net", "unet_full", "best"),
    ("Cascade", "cascade_full", "best"),
    ("U-Net + D", "unet_gan", "latest"),
    ("Cascade + D", "cascade_gan", "latest"),
    ("U-Net + power", "unet_power", "latest"),
    ("U-Net + weak D + power", "unet_gan_power_lo", "latest"),   # recommended realism model
    ("U-Net amplitude-only", "unet_magnitude", "best"),
]


def load(run: str, ckpt: str, device):
    path = Path("runs") / run / f"{ckpt}.pt"
    if not path.exists():
        return None, None
    cfg = yaml.safe_load((Path("runs") / run / "config.yaml").read_text())
    model = build_model(cfg["model"]).to(device).eval()
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in ck["ema"].items() if k != "n_averaged"})
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="runs/figures")
    p.add_argument("--volume", default=None, help="prepared volume name, e.g. phase__NailB")
    p.add_argument("--y", type=int, default=None, help="B-scan index within --volume")
    args = p.parse_args()
    if (args.volume is None) != (args.y is None):
        p.error("--volume and --y go together")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models = []
    for label, run, ck in MODELS:
        model, cfg = load(run, ck, device)
        if model is None:
            print(f"skip {label}: runs/{run}/{ck}.pt missing")
            continue
        models.append((label, model, cfg))
    cfg0 = models[0][2]
    K = cfg0["data"]["factor"]
    assert all(c["data"]["factor"] == K for _, _, c in models), "all models must share K"
    divisor = math.lcm(max(m.divisor for _, m, _ in models), K)
    ds = EvalBScans(args.data_root or cfg0["data"]["root"], "test", per_volume=None, divisor=divisor,
                    sources=cfg0["data"].get("sources"))

    if args.volume is not None:
        if (args.volume, args.y) not in ds.items:
            raise SystemExit(f"{args.volume} y={args.y} is not a test B-scan; test volumes: "
                             f"{sorted({n for n, _ in ds.items})}")
        i = ds.items.index((args.volume, args.y))
        sample = ds.vols.volume(args.volume)["group"].split("/", 1)[-1]
        jobs = [(sample, i, f"recon_{sample}_{args.volume.split('__')[-1]}_y{args.y}.npz")]
    else:
        first = {}
        for i, (name, y) in enumerate(ds.items):
            first.setdefault(ds.vols.volume(name)["group"].split("/", 1)[-1], []).append(i)
        jobs = []
        for sample, idx in sorted(first.items()):
            vol = ds.items[idx[0]][0]
            idx = [i for i in idx if ds.items[i][0] == vol]
            jobs.append((sample, idx[len(idx) // 2], f"recon_{sample}.npz"))
    dtype = amp_dtype(cfg0["train"].get("precision", "auto"))
    for sample, i, fname in jobs:
        x, name, y = ds[i]
        x = x.to(device)[None]
        x_meas = measure(x, K, 0)
        fields = {"Ground truth": x[0], "Interpolation": x_meas[0]}
        with torch.no_grad():
            for label, model, _ in models:
                with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None and x.is_cuda):
                    fields[label] = model(x_meas, K, 0).to(torch.complex64)[0]
        labels = list(fields)
        np.savez_compressed(out / fname, labels=np.array(labels), sample=sample, volume=name, y=y,
                            factor=K, **{f"f{i}": fields[lab].cpu().numpy() for i, lab in enumerate(labels)})
        print(f"{sample}: {name} y={y}: {', '.join(labels)} -> {out / fname}")


if __name__ == "__main__":
    main()
