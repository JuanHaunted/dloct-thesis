"""
Write a small synthetic complex OCT volume for smoke tests (not for results).

Fully developed speckle (circular complex Gaussian scatterers) through a Gaussian lateral PSF
and axial coherence envelope, with a few layers of varying reflectivity and one "flow" band
carrying a constant inter-A-line phase shift (a Doppler signal). Saved in the raw layout
(Z, X, Y, 2) float32 so it exercises ``dloct.prepare_data``.

    uv run python scripts/make_fake_data.py --out data/train/synthetic
"""

import argparse
from pathlib import Path

import numpy as np


def make_volume(z=256, x=256, y=48, lateral_hw=0.22, seed=0):
    rng = np.random.default_rng(seed)
    field = (rng.standard_normal((z, x, y)) + 1j * rng.standard_normal((z, x, y))) / np.sqrt(2)

    depth = np.arange(z)[:, None, None]
    layers = np.zeros((z, 1, 1))
    for top, bottom, r in ((40, 70, 1.0), (70, 120, 0.3), (120, 180, 0.6), (180, 230, 0.15)):
        layers[top:bottom] = r
    field *= layers * np.exp(-depth / 400)

    # Lateral Gaussian PSF: spectral half-width at 1 % of peak = lateral_hw (normalized).
    sigma = lateral_hw / np.sqrt(2 * np.log(100))
    for axis, n in ((1, x), (2, y)):
        f = np.fft.fftfreq(n)
        shape = [1, 1, 1]
        shape[axis] = n
        field = np.fft.ifft(np.fft.fft(field, axis=axis) * np.exp(-f ** 2 / (2 * sigma ** 2)).reshape(shape), axis=axis)
    fz = np.fft.fftfreq(z)
    field = np.fft.ifft(np.fft.fft(field, axis=0) * np.exp(-fz ** 2 / (2 * 0.15 ** 2))[:, None, None], axis=0)

    # Flow band: constant phase step between adjacent A-lines along the fast axis.
    flow = np.zeros((z, 1, 1), dtype=bool)
    flow[140:160] = True
    field = np.where(flow, field * np.exp(1j * 0.8 * np.arange(x))[None, :, None], field)

    field += 0.003 * (rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape))
    return np.stack([field.real, field.imag], axis=-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/train/synthetic")
    p.add_argument("--n", type=int, default=3, help="number of distinct fake samples")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        vol = make_volume(seed=i)
        path = out / f"polInt1_polOut1_fake{i}.npy"
        np.save(path, vol)
        print(f"wrote {path} {vol.shape}")


if __name__ == "__main__":
    main()
