"""Comparison figures: amplitude (dB), phase, and inter-A-line phase difference per method."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .losses import lateral_phasor
from .metrics import tissue_mask


def _np(z):
    return z.detach().cpu().numpy()


def comparison_figure(gt: torch.Tensor, preds: dict, path, tissue_db=-30.0, dyn_range=50.0,
                      title=None):
    """
    ``gt`` is a complex (Z, X) B-scan, ``preds`` maps method name -> complex (Z, X).
    Rows: GT then each method. Columns: amplitude dB, phase (tissue only), Doppler Δφ
    (tissue only), amplitude error vs GT.
    """
    rows = {"ground truth": gt, **preds}
    mask = _np(tissue_mask(gt[None], tissue_db)[0])
    mask_d = mask[:, 1:] & mask[:, :-1]
    gt_db = 20 * np.log10(np.abs(_np(gt)) + 1e-12)

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.2 * len(rows)), squeeze=False)
    for r, (name, z) in enumerate(rows.items()):
        zn = _np(z)
        db = 20 * np.log10(np.abs(zn) + 1e-12)
        phase = np.where(mask, np.angle(zn), np.nan)
        dphi = np.where(mask_d, np.angle(_np(lateral_phasor(z[None])[0])), np.nan)
        err = np.clip(db - gt_db, -20, 20)
        panels = [
            (db, dict(cmap="gray", vmin=-dyn_range, vmax=0), "amplitude [dB]"),
            (phase, dict(cmap="twilight", vmin=-np.pi, vmax=np.pi), "phase [rad]"),
            (dphi, dict(cmap="twilight", vmin=-np.pi, vmax=np.pi), "inter-A-line Δφ [rad]"),
            (err, dict(cmap="RdBu_r", vmin=-20, vmax=20), "amplitude error [dB]"),
        ]
        for c, (img, kw, label) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, aspect="auto", interpolation="nearest", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(name)
            fig.colorbar(im, ax=ax, fraction=0.04)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mps_figure(spectra: dict, path, factor: int):
    """Lateral mean power spectra (dB) per method; ``spectra`` maps name -> (freq, mps)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (f, p) in spectra.items():
        ax.plot(f, 10 * np.log10(p / p.max() + 1e-12), label=name, lw=1.4)
    for s in (-1, 1):
        ax.axvline(s * 0.5 / factor, color="k", ls=":", lw=1)
    ax.set_xlabel("normalized lateral frequency")
    ax.set_ylabel("mean power spectrum [dB]")
    ax.set_ylim(-60, 2)
    ax.legend()
    ax.set_title(f"Lateral MPS (dotted: measured band edge, K={factor})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
