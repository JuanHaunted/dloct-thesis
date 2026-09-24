"""
Datasets over prepared complex OCT volumes.

Raw tomograms are ``(Z, X, Y[, 2])`` (depth, fast lateral, slow lateral, Re/Im). Reading one
B-scan ``tom[:, :, y]`` from that layout touches the whole file, so ``dloct.prepare_data``
converts each volume once to a contiguous complex64 array of shape ``(Y, Z, X)`` and records
its normalization scale and split in ``meta.json``. Everything here reads that format.

Tensors returned are complex64 ``(Z, X)`` B-scans divided by the per-volume scale (99.9th
percentile amplitude), lateral axis last.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedVolumes:
    """Memory-mapped view of ``<root>/meta.json`` and the ``.npy`` volumes it lists."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found; run `python -m dloct.prepare_data` first")
        self.meta = json.loads(meta_path.read_text())
        self._arrays = {}

    def array(self, name: str) -> np.ndarray:
        # Opened lazily so each DataLoader worker gets its own memmap.
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def volume(self, name: str) -> dict:
        return self.meta["volumes"][name]

    def ranges(self, split: str, sources=None) -> list[tuple[str, int, int]]:
        """(volume, y_start, y_end) B-scan ranges of ``split``, optionally only from ``sources``."""
        return [tuple(r) for r in self.meta["splits"][split]
                if sources is None or self.volume(r[0])["source"] in sources]


def _crop_to_multiple(n: int, m: int) -> int:
    return (n // m) * m


class TrainPatches(Dataset):
    """
    Random (patch_z, patch_x) crops of random training B-scans, with random global phase and
    lateral flip. Crops mostly-background patches: up to ``tries`` candidates are drawn and
    the first with mean intensity above ``min_energy`` (normalized units) is kept, else the
    most energetic one.

    Items are generated from ``(seed, rank, index)`` so every rank and worker draws
    different, reproducible patches without a sampler.
    """

    def __init__(self, root, patch=(256, 256), length=10 ** 7, seed=0, rank=0,
                 min_energy=1e-3, tries=8, divisor=16, sources=None):
        self.vols = PreparedVolumes(root)
        self.ranges = self.vols.ranges("train", sources)
        if not self.ranges:
            raise ValueError("empty train split")
        sizes = np.array([e - s for _, s, e in self.ranges], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.patch = patch
        self.length = length
        self.seed, self.rank = seed, rank
        self.min_energy, self.tries = min_energy, tries
        self.divisor = divisor

    def __len__(self):
        return self.length

    def _draw(self, rng):
        name, y0, y1 = self.ranges[rng.choice(len(self.ranges), p=self.weights)]
        arr = self.vols.array(name)
        _, z_len, x_len = arr.shape
        pz = min(self.patch[0], _crop_to_multiple(z_len, self.divisor))
        px = min(self.patch[1], _crop_to_multiple(x_len, self.divisor))
        y = rng.integers(y0, y1)
        z = rng.integers(0, z_len - pz + 1)
        x = rng.integers(0, x_len - px + 1)
        crop = np.asarray(arr[y, z:z + pz, x:x + px]) / self.vols.volume(name)["scale"]
        return crop

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.rank, index])
        best, best_e = None, -1.0
        for _ in range(self.tries):
            crop = self._draw(rng)
            e = float(np.mean(np.abs(crop) ** 2))
            if e > best_e:
                best, best_e = crop, e
            if e >= self.min_energy:
                break
        crop = best * np.exp(1j * rng.uniform(0, 2 * math.pi))
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(crop, dtype=np.complex64))


class EvalBScans(Dataset):
    """
    Full B-scans from a split, ``per_volume`` of them evenly spaced over each range (all of
    them if ``per_volume`` is None). Z and X are cropped to multiples of ``divisor``.
    Returns ``(bscan, volume_name, y)``.
    """

    def __init__(self, root, split="val", per_volume=8, divisor=16, sources=None):
        self.vols = PreparedVolumes(root)
        self.items = []
        for name, y0, y1 in self.vols.ranges(split, sources):
            ys = range(y0, y1)
            if per_volume is not None and len(ys) > per_volume:
                ys = np.linspace(y0, y1 - 1, per_volume).round().astype(int)
            self.items += [(name, int(y)) for y in ys]
        self.divisor = divisor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, y = self.items[i]
        arr = self.vols.array(name)
        _, z_len, x_len = arr.shape
        z_len, x_len = _crop_to_multiple(z_len, self.divisor), _crop_to_multiple(x_len, self.divisor)
        b = np.asarray(arr[y, :z_len, :x_len]) / self.vols.volume(name)["scale"]
        return torch.from_numpy(b.astype(np.complex64)), name, y
