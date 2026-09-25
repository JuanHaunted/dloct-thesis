"""
Convert raw tomograms to the training format read by ``dloct.data``.

For every ``*.npy`` under ``--src`` (recursively):
  * interpret it as ``(Z, X, Y)`` complex or ``(Z, X, Y, 2)`` Re/Im (axis order set by
    ``--layout``),
  * for sources listed in ``--bulk-phase`` (default: ``phase``, the real acquisitions), remove
    the bulk phase between adjacent A-lines (see ``remove_bulk_phase``),
  * write ``<out>/<name>.npy`` as contiguous complex64 ``(Y, Z, X)`` (B-scan major),
  * record the normalization scale (99.9th percentile amplitude) and lateral-coherence
    diagnostics in ``<out>/meta.json``.

Then it assigns splits **by sample**, never by B-scan: polarization channels of the same
acquisition (``polInt1_polOut2_foo`` and ``polInt2_polOut1_foo``, or the ``A``/``B`` channels
``Fovea1A`` and ``Fovea1B``) share a group and always land in the same split. If a source directory holds fewer than three groups, the split
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
# Detection channel letter A/B, at the end ("Fovea1A") or before a New/Old tag ("OpticNerveANew").
_CHANNEL = re.compile(r"(?<=[0-9a-z])[AB](?=(?:New|Old)?$)")


def group_of(source: str, stem: str) -> str:
    base = _CHANNEL.sub("", _POL.sub("", stem)) or stem
    return f"{source}/{base}"


def remove_bulk_phase(vol: np.ndarray) -> np.ndarray:
    """
    Remove the random bulk phase between adjacent A-lines of each B-scan, in place.

    Real acquisitions carry a per-A-line phase offset (sample motion, trigger jitter) that is
    unpredictable from neighbouring A-lines and irrelevant for functional imaging: Doppler/OCE
    remove it too. The step between A-lines x and x+1 is the phase of Σ_z T(z,x+1)·T*(z,x)
    (intensity-weighted, so dominated by tissue); its cumulative sum along x is removed.
    ``vol`` is (Y, Z, X).
    """
    for y in range(vol.shape[0]):
        b = vol[y]
        step = np.angle((b[:, 1:] * b[:, :-1].conj()).sum(axis=0))
        phi = np.concatenate([[0.0], np.cumsum(step)])
        vol[y] = b * np.exp(-1j * phi)[None, :].astype(np.complex64)
    return vol


def lateral_diagnostics(vol: np.ndarray, n_bscans: int = 16) -> dict:
    """
    Complex correlation between A-lines ``lag`` apart (tissue: top 30 % amplitude) and the
    fraction of lateral spectral energy inside the band a K-fold decimation keeps. Low
    correlation at lag K means the missing A-lines are barely predictable from the measured ones.
    """
    b = vol[np.linspace(0, vol.shape[0] - 1, min(n_bscans, vol.shape[0])).astype(int)]
    thr = np.percentile(np.abs(b), 70)
    out = {}
    for lag in (1, 2, 4):
        p, q = b[..., lag:], b[..., :-lag]
        m = (np.abs(p) > thr) & (np.abs(q) > thr)
        num = np.abs((p * q.conj())[m].sum())
        out[f"rho_lag{lag}"] = float(num / np.sqrt((np.abs(p[m]) ** 2).sum() * (np.abs(q[m]) ** 2).sum()))
    s = (np.abs(np.fft.fft(b, axis=-1)) ** 2).mean(axis=(0, 1))
    f = np.fft.fftfreq(b.shape[-1])
    for k in (2, 4):
        out[f"inband_energy_K{k}"] = float(s[np.abs(f) < 0.5 / k].sum() / s.sum())
    return out


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


def assign_splits(volumes: dict, split_file: str | None = None) -> dict:
    if split_file:
        return splits_from_file(volumes, split_file)
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


def splits_from_file(volumes: dict, split_file: str) -> dict:
    """
    Explicit split by sample: YAML ``{val: [...], test: [...]}`` listing group names without
    the source prefix (e.g. ``Fovea5``, ``OpticNerve4``). Every other sample goes to train.
    """
    import yaml

    spec = yaml.safe_load(Path(split_file).read_text()) or {}
    by_group = {}
    for name, v in volumes.items():
        by_group.setdefault(v["group"].split("/", 1)[1], []).append(name)
    unknown = [g for role in ("val", "test") for g in spec.get(role, []) if g not in by_group]
    if unknown:
        raise SystemExit(f"{split_file}: unknown samples {unknown}; known: {sorted(by_group)}")
    role_of = {g: role for role in ("val", "test") for g in spec.get(role, [])}
    splits = {"train": [], "val": [], "test": []}
    for group, names in sorted(by_group.items()):
        for name in sorted(names):
            splits[role_of.get(group, "train")].append([name, 0, volumes[name]["shape"][0]])
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/train")
    p.add_argument("--out", default="data/prepared")
    p.add_argument("--layout", default="ZXY", help="axis order of the raw volume (default ZXY)")
    p.add_argument("--bulk-phase", nargs="*", default=["phase"], metavar="SOURCE",
                   help="sources (subdirectories) to bulk-phase correct; default: phase")
    p.add_argument("--split-file", default=None,
                   help="YAML {val: [...], test: [...]} of sample names; default: automatic")
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
        diag = {"raw": lateral_diagnostics(vol)}
        corrected = source in args.bulk_phase
        if corrected:
            remove_bulk_phase(vol)
            diag["bulk_corrected"] = lateral_diagnostics(vol)
        scale = amplitude_scale(vol)
        np.save(out / f"{name}.npy", vol)
        volumes[name] = dict(source=source, group=group_of(source, path.stem), file=str(path),
                             shape=list(vol.shape), scale=scale, bulk_phase_corrected=corrected,
                             diagnostics=diag)
        print(f"{name}: (Y,Z,X)={vol.shape} scale={scale:.4g} group={volumes[name]['group']}")
        for tag, d in diag.items():
            print(f"    {tag:15s} " + " ".join(f"{k}={v:.3f}" for k, v in d.items()))
        del vol

    meta = dict(volumes=volumes, splits=assign_splits(volumes, args.split_file), layout="YZX complex64")
    meta_path.write_text(json.dumps(meta, indent=2))
    for split, ranges in meta["splits"].items():
        n = sum(e - s for _, s, e in ranges)
        samples = sorted({volumes[name]["group"].split("/", 1)[1] for name, _, _ in ranges})
        print(f"{split}: {len(ranges)} volumes, {n} B-scans: {', '.join(samples)}")


if __name__ == "__main__":
    main()
