"""
visualizer.py
-------------
Publication-quality plots for all CWT result types and the singularity spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from typing import Optional, List, Tuple, Dict
import os


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f8f8",
        "axes.grid":        True,
        "grid.color":       "white",
        "grid.linewidth":   0.8,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
        "font.size":        10,
    })


CMAPS = {
    "power":    "plasma",
    "phase":    "hsv",
    "real":     "RdBu_r",
    "spectrum": "inferno",
}


# ---------------------------------------------------------------------------
# 1-D CWT
# ---------------------------------------------------------------------------

def plot_cwt_1d(cwt_result,
                signal: Optional[np.ndarray] = None,
                title: str = "1-D CWT",
                log_power: bool = True,
                show_coi: bool = True,
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Scalogram + (optionally) original signal.

    Parameters
    ----------
    cwt_result : CWTResult1D
    signal     : original signal for top panel (optional)
    log_power  : use log scale for colour map
    show_coi   : overlay cone of influence boundary
    """
    _apply_style()
    has_signal = signal is not None
    nrows = 3 if has_signal else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3 * nrows),
                              gridspec_kw={"height_ratios": [1, 2, 1.2][:nrows]})

    scales = cwt_result.scales
    freqs = cwt_result.frequencies
    N = cwt_result.signal_length
    dt = cwt_result.dt
    t = np.arange(N) * dt
    power = cwt_result.power  # (n_scales, N)

    ax_idx = 0
    if has_signal:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(t, signal, color="#1a6fc4", lw=1.2)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=12, fontweight="bold")

    # --- Scalogram
    ax = axes[ax_idx]; ax_idx += 1
    norm = LogNorm(vmin=power.max() * 1e-5, vmax=power.max()) if log_power else Normalize()
    im = ax.pcolormesh(t, np.log2(scales), power,
                        cmap=CMAPS["power"], norm=norm, shading="auto")
    plt.colorbar(im, ax=ax, label="Power" + (" (log)" if log_power else ""))

    # Cone of influence
    if show_coi:
        e_folding = np.sqrt(2.0)  # for Morlet; approximate for others
        coi = e_folding * scales
        coi_left  = np.log2(coi)
        coi_right = np.log2(coi)
        t_left  = coi
        t_right = t[-1] - coi
        valid = t_left < t_right
        if valid.any():
            ax.fill_betweenx(np.log2(scales)[valid],
                             t[0], t_left[valid],
                             alpha=0.3, color="gray", label="COI")
            ax.fill_betweenx(np.log2(scales)[valid],
                             t_right[valid], t[-1],
                             alpha=0.3, color="gray")

    # Secondary y-axis with frequency labels
    ax.set_ylabel("Scale (log₂)")
    ax.set_xlim(t[0], t[-1])
    yticks = np.log2(scales[::max(1, len(scales) // 8)])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{2**y:.1f}" for y in yticks])
    if not has_signal:
        ax.set_title(title, fontsize=12, fontweight="bold")

    # --- Global wavelet spectrum
    ax = axes[ax_idx]
    global_spectrum = power.mean(axis=1)
    ax.plot(freqs, global_spectrum, color="#c44e0a", lw=1.5)
    ax.fill_between(freqs, global_spectrum, alpha=0.25, color="#c44e0a")
    ax.set_xlabel("Pseudo-frequency")
    ax.set_ylabel("Mean Power")
    ax.set_title("Global Wavelet Spectrum")
    ax.set_xlim(freqs.min(), freqs.max())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2-D CWT
# ---------------------------------------------------------------------------

def plot_cwt_2d(cwt_result,
                image: Optional[np.ndarray] = None,
                scale_indices: Optional[List[int]] = None,
                title: str = "2-D CWT",
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Show the power at selected scales side-by-side with the original image.
    """
    _apply_style()
    scales = cwt_result.scales
    power = cwt_result.power  # (n_scales, H, W)

    if scale_indices is None:
        n = min(4, len(scales))
        scale_indices = np.linspace(0, len(scales) - 1, n, dtype=int).tolist()

    n_show = len(scale_indices)
    ncols = n_show + (1 if image is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    col = 0
    if image is not None:
        axes[col].imshow(image, cmap="gray", aspect="auto")
        axes[col].set_title("Original Image")
        axes[col].axis("off")
        col += 1

    for si in scale_indices:
        axes[col].imshow(power[si], cmap=CMAPS["power"],
                          norm=LogNorm(vmin=power[si].max() * 1e-5 + 1e-30,
                                       vmax=power[si].max() + 1e-30),
                          aspect="auto")
        axes[col].set_title(f"Scale {scales[si]:.2f}")
        axes[col].axis("off")
        col += 1

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3-D + Time CWT
# ---------------------------------------------------------------------------

def plot_cwt_3d_time(cwt_result,
                      time_indices: Optional[List[int]] = None,
                      scale_index: int = 0,
                      slice_dim: str = "z",
                      slice_idx: Optional[int] = None,
                      title: str = "3-D+Time CWT",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Show power slices for selected time frames at a given scale.

    Parameters
    ----------
    cwt_result   : CWTResult3DTime, coefficients shape (T, n_scales, D, H, W)
    time_indices : which time frames to plot (default: 4 evenly spaced)
    scale_index  : which scale to display
    slice_dim    : 'x', 'y', or 'z' — dimension to slice through
    slice_idx    : index along slice_dim (default: middle)
    """
    _apply_style()
    T, n_scales, D, H, W = cwt_result.coefficients.shape

    if time_indices is None:
        time_indices = np.linspace(0, T - 1, min(4, T), dtype=int).tolist()

    dim_map = {"z": (slice(None), slice(None)), "y": None, "x": None}

    fig, axes = plt.subplots(1, len(time_indices),
                              figsize=(4 * len(time_indices), 4))
    if len(time_indices) == 1:
        axes = [axes]

    mid_d = D // 2 if slice_idx is None else slice_idx
    mid_h = H // 2 if slice_idx is None else slice_idx
    mid_w = W // 2 if slice_idx is None else slice_idx

    for col, ti in enumerate(time_indices):
        vol_power = cwt_result.power[ti, scale_index]  # (D, H, W)
        if slice_dim == "z":
            slc = vol_power[mid_d]
        elif slice_dim == "y":
            slc = vol_power[:, mid_h, :]
        else:
            slc = vol_power[:, :, mid_w]

        axes[col].imshow(slc, cmap=CMAPS["power"], aspect="auto")
        axes[col].set_title(f"t={ti}, s={cwt_result.scales[scale_index]:.2f}")
        axes[col].axis("off")

    fig.suptitle(f"{title} — scale index {scale_index}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Multi-band CWT
# ---------------------------------------------------------------------------

def plot_cwt_multiband(cwt_result,
                        scale_index: int = 0,
                        band_images: Optional[Dict[str, np.ndarray]] = None,
                        title: str = "Multi-band CWT",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Show original bands (top row) and CWT power at a given scale (bottom row).
    """
    _apply_style()
    bands = cwt_result.band_names
    n = len(bands)
    nrows = 2 if band_images is not None else 1
    fig, axes = plt.subplots(nrows, n, figsize=(4 * n, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, band in enumerate(bands):
        if band_images is not None:
            axes[0, col].imshow(band_images[band], cmap="gray", aspect="auto")
            axes[0, col].set_title(f"Band: {band}")
            axes[0, col].axis("off")

        pwr = cwt_result.power[band][scale_index]
        im = axes[nrows - 1, col].imshow(
            pwr, cmap=CMAPS["power"],
            norm=LogNorm(vmin=pwr.max() * 1e-5 + 1e-30, vmax=pwr.max() + 1e-30),
            aspect="auto")
        axes[nrows - 1, col].set_title(
            f"{band} power\n(scale {cwt_result.scales[scale_index]:.2f})")
        axes[nrows - 1, col].axis("off")
        plt.colorbar(im, ax=axes[nrows - 1, col])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Singularity spectrum
# ---------------------------------------------------------------------------

def plot_singularity_spectrum(ss,
                               holder: Optional[object] = None,
                               title: str = "Singularity Spectrum",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Three-panel plot:
      1. τ(q)  — scaling exponents
      2. D(α)  — singularity spectrum
      3. Pointwise Hölder exponents (if holder is provided)

    Parameters
    ----------
    ss     : SingularitySpectrum
    holder : HolderField (optional) for pointwise α(t)
    """
    _apply_style()
    has_holder = holder is not None
    ncols = 3 if has_holder else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Panel 1: τ(q)
    ax = axes[0]
    ax.plot(ss.q_values, ss.tau, color="#1a6fc4", lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("q")
    ax.set_ylabel("τ(q)")
    ax.set_title("Scaling exponents τ(q)")

    # Panel 2: D(α)
    ax = axes[1]
    # Colour by q (warm → large q)
    cmap_q = plt.get_cmap("coolwarm")
    q_norm = (ss.q_values - ss.q_values.min()) / (np.ptp(ss.q_values) + 1e-30)
    valid = len(ss.alpha)
    if valid > 0:
        ax.plot(ss.alpha, ss.D_alpha, color="#c44e0a", lw=2, zorder=2)
        ax.fill_between(ss.alpha, 0, ss.D_alpha, alpha=0.15, color="#c44e0a")
        ax.axvline(ss.alpha_max, color="#c44e0a", ls="--", lw=1,
                   label=f"α_max = {ss.alpha_max:.3f}")
        ax.axvline(ss.hurst, color="#1a6fc4", ls="--", lw=1,
                   label=f"H ≈ {ss.hurst:.3f}")
    ax.set_xlabel("α (Hölder exponent)")
    ax.set_ylabel("D(α)")
    ax.set_title(f"Singularity Spectrum\nWidth = {ss.width:.3f}")
    ax.legend(fontsize=9)

    # Panel 3: pointwise Hölder
    if has_holder:
        ax = axes[2]
        sc = ax.scatter(holder.t, holder.alpha, c=holder.alpha,
                        cmap="plasma", s=4, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="α(t)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Local Hölder exponent α(t)")
        ax.set_title("Pointwise Hölder Exponents")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_multiscale_entropy(scales: np.ndarray,
                             entropy: np.ndarray,
                             title: str = "Multiscale Rényi Entropy",
                             save_path: Optional[str] = None) -> plt.Figure:
    """Simple plot of scale-resolved entropy."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(scales, entropy, color="#1a6fc4", lw=2, marker="o", ms=4)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
