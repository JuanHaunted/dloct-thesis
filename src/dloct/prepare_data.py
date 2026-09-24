"""
Convert raw tomograms to the training format read by ``dloct.data``.

For every ``*.npy`` under ``--src`` (recursively):
  * interpret it as ``(Z, X, Y)`` complex or ``(Z, X, Y, 2)`` Re/Im (axis order set by
    ``--layout``),
  * write ``<out>/<name>.npy`` as contiguous complex64 ``(Y, Z, X)`` (B-scan major),
  * record the normalization scale (99.9th percentile amplitude) in ``<out>/meta.json``.

Then it assigns splits **by sample**, never by B-scan: polarization channels of the same
acquisition (``polInt1_polOut2_foo`` and ``polInt2_polOut1_foo``) share a group and always
land in the same split. If a source directory holds fewer than three groups, the split
falls back to disjoint B-scan ranges with gaps (weaker; fine for local smoke tests).

Usage:
    python -m dloct.prepare_data --src data/train --out data/prepared
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

_POL = re.compile(r"polInt\d+_polOut\d+_?", re.IGNORECASE)


def group_of(source: str, stem: str) -> str:
    return f"{source}/{_POL.sub('', stem) or stem}"


def load_complex(path: Path, layout: str) -> np.ndarray:
    raw = np.load(path, mmap_mode="r")
    if np.iscomplexobj(raw):
        vol = raw
    elif raw.ndim == 4 and raw.shape[-1] == 2:
        vol = raw[..., 0] + 1j * raw[..., 1]
    else:
        raise ValueError(f"{path}: expected complex or trailing Re/Im axis, got {raw.shape} {raw.dtype}")
    if vol.ndim != 3:
        raise ValueError(f"{path}: expected 3 spatial axes, got {vol.shape}")
    order = [layout.index(a) for a in "YZX"]
    return np.ascontiguousarray(np.transpose(vol, order), dtype=np.complex64)


def amplitude_scale(vol: np.ndarray, q: float = 99.9, n: int = 2_000_000, seed: int = 0) -> float:
    flat = vol.reshape(-1)
    idx = np.random.default_rng(seed).integers(0, flat.size, min(n, flat.size))
    return float(np.percentile(np.abs(flat[idx]), q))


def assign_splits(volumes: dict) -> dict:
    splits = {"train": [], "val": [], "test": []}
    by_source = {}
    for name, v in volumes.items():
        by_source.setdefault(v["source"], {}).setdefault(v["group"], []).append(name)
    for source, groups in sorted(by_source.items()):
        keys = sorted(groups)
        if len(keys) >= 3:
            n_hold = max(1, round(0.15 * len(keys)))
            roles = {k: "train" for k in keys}
            for k in keys[-n_hold:]:
                roles[k] = "test"
            for k in keys[-2 * n_hold:-n_hold]:
                roles[k] = "val"
            for k, role in roles.items():
                for name in groups[k]:
                    splits[role].append([name, 0, volumes[name]["shape"][0]])
        else:
            # Too few samples: split each volume along the slow axis with gaps.
            for k in keys:
                for name in groups[k]:
                    ny = volumes[name]["shape"][0]
                    cut = lambda f: int(round(f * ny))
                    splits["train"].append([name, 0, cut(0.70)])
                    splits["val"].append([name, cut(0.75), cut(0.85)])
                    splits["test"].append([name, cut(0.90), ny])
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/train")
    p.add_argument("--out", default="data/prepared")
    p.add_argument("--layout", default="ZXY", help="axis order of the raw volume (default ZXY)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    volumes = json.loads(meta_path.read_text())["volumes"] if meta_path.exists() else {}

    files = sorted(src.rglob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy files under {src}")
    for path in files:
        source = path.parent.relative_to(src).as_posix() or "."
        name = f"{source}__{path.stem}".replace("/", "__").lstrip("._")
        if name in volumes and (out / f"{name}.npy").exists() and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        vol = load_complex(path, args.layout)
        scale = amplitude_scale(vol)
        np.save(out / f"{name}.npy", vol)
        volumes[name] = dict(source=source, group=group_of(source, path.stem), file=str(path),
                             shape=list(vol.shape), scale=scale)
        print(f"{name}: (Y,Z,X)={vol.shape} scale={scale:.4g} group={volumes[name]['group']}")
        del vol

    meta = dict(volumes=volumes, splits=assign_splits(volumes), layout="YZX complex64")
    meta_path.write_text(json.dumps(meta, indent=2))
    for split, ranges in meta["splits"].items():
        n = sum(e - s for _, s, e in ranges)
        print(f"{split}: {len(ranges)} ranges, {n} B-scans")


if __name__ == "__main__":
    main()
