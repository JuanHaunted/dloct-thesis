"""
================================================================================
RIGOROUS GABOR TRANSFORM ANALYSIS PIPELINE
================================================================================
Supports:
  - 1D signals (arbitrary length)
  - 2D signals (arbitrary spatial dimensions H x W)
  - 2D multi-band signals (H x W x L bands)

Author: Generated Pipeline
Dependencies: numpy, scipy, matplotlib, scikit-image, pandas
================================================================================
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaborParams1D:
    """Parameters for 1D Gabor filterbank."""
    frequencies: List[float]          # Center frequencies (normalized 0–1, where 1 = Nyquist)
    sigmas: List[float]               # Gaussian envelope widths (in samples)
    n_cycles: Optional[int] = None    # Alternative: define sigma by number of cycles
    window_size: Optional[int] = None # Filter kernel size (auto if None)
    padding: str = 'reflect'          # 'reflect', 'wrap', 'constant'


@dataclass
class GaborParams2D:
    """Parameters for 2D Gabor filterbank."""
    frequencies: List[float]          # Spatial frequencies (cycles/pixel)
    thetas: List[float]               # Orientations in radians
    sigmas_x: List[float]             # Gaussian width along x-axis
    sigmas_y: List[float]             # Gaussian width along y-axis
    n_stds: float = 3.0               # Kernel size = 2 * n_stds * max_sigma + 1
    padding: str = 'reflect'          # Padding mode


@dataclass
class PipelineConfig:
    """Global pipeline configuration."""
    output_dir: str = './gabor_output'
    save_figures: bool = True
    save_data: bool = True
    verbose: bool = True
    fig_dpi: int = 150
    fig_format: str = 'png'
    compute_statistics: bool = True
    compute_energy: bool = True
    compute_phase: bool = True


@dataclass
class AnalysisResult1D:
    """Container for 1D Gabor analysis results."""
    signal_length: int
    n_filters: int
    frequencies: np.ndarray
    sigmas: np.ndarray
    responses_real: np.ndarray        # Shape: (n_filters, signal_length)
    responses_imag: np.ndarray
    responses_magnitude: np.ndarray
    responses_phase: np.ndarray
    energy_per_filter: np.ndarray
    statistics: Dict[str, Any] = field(default_factory=dict)
    processing_time_s: float = 0.0


@dataclass
class AnalysisResult2D:
    """Container for 2D Gabor analysis results."""
    spatial_shape: Tuple[int, int]
    n_filters: int
    n_frequencies: int
    n_orientations: int
    frequencies: np.ndarray
    thetas: np.ndarray
    responses_magnitude: np.ndarray   # Shape: (n_freq, n_theta, H, W)
    responses_phase: np.ndarray
    energy_map: np.ndarray            # Shape: (H, W)  — integrated over all filters
    dominant_orientation: np.ndarray  # Shape: (H, W)
    dominant_frequency: np.ndarray    # Shape: (H, W)
    statistics: Dict[str, Any] = field(default_factory=dict)
    processing_time_s: float = 0.0


@dataclass
class AnalysisResultMultiBand:
    """Container for multi-band 2D Gabor analysis results."""
    spatial_shape: Tuple[int, int]
    n_bands: int
    band_results: List[AnalysisResult2D] = field(default_factory=list)
    inter_band_correlation: Optional[np.ndarray] = None  # (n_bands, n_bands)
    fused_energy_map: Optional[np.ndarray] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    processing_time_s: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GABOR KERNEL CONSTRUCTORS
# ─────────────────────────────────────────────────────────────────────────────

class GaborKernel1D:
    """
    1D Gabor kernel: g(t) = exp(-t²/2σ²) · exp(i·2π·f·t)
    Real part: cosine-modulated Gaussian
    Imaginary part: sine-modulated Gaussian
    """

    @staticmethod
    def build(frequency: float, sigma: float,
              size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a 1D Gabor filter kernel (real and imaginary parts).

        Parameters
        ----------
        frequency : float
            Normalized frequency [0, 0.5] (fraction of sampling rate)
        sigma : float
            Standard deviation of Gaussian envelope in samples
        size : int, optional
            Kernel length. Defaults to 6*sigma + 1 (covers ±3σ)

        Returns
        -------
        real_kernel, imag_kernel : np.ndarray
        """
        if size is None:
            size = int(6 * sigma + 1)
            if size % 2 == 0:
                size += 1  # ensure odd length for symmetric kernel

        half = size // 2
        t = np.arange(-half, half + 1, dtype=np.float64)

        gaussian = np.exp(-0.5 * (t / sigma) ** 2)
        # Normalize Gaussian to unit energy
        gaussian /= (gaussian.sum() + 1e-12)

        omega = 2.0 * np.pi * frequency
        real_kernel = gaussian * np.cos(omega * t)
        imag_kernel = gaussian * np.sin(omega * t)

        # DC correction: subtract mean so DC response is zero
        real_kernel -= real_kernel.mean()

        return real_kernel, imag_kernel

    @staticmethod
    def from_cycles(frequency: float, n_cycles: int = 5,
                    size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Build kernel from a number of cycles (sigma = n_cycles / (2π·f))."""
        if frequency < 1e-10:
            raise ValueError("Frequency must be positive.")
        sigma = n_cycles / (2.0 * np.pi * frequency)
        return GaborKernel1D.build(frequency, sigma, size)


class GaborKernel2D:
    """
    2D Gabor kernel:
    g(x,y) = exp(-0.5*(x'²/σx² + y'²/σy²)) · exp(i·2π·f·x')
    where x' = x·cos(θ) + y·sin(θ), y' = -x·sin(θ) + y·cos(θ)
    """

    @staticmethod
    def build(frequency: float, theta: float,
              sigma_x: float, sigma_y: float,
              n_stds: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a 2D Gabor filter kernel.

        Parameters
        ----------
        frequency : float
            Spatial frequency in cycles/pixel
        theta : float
            Orientation in radians (0 = horizontal)
        sigma_x : float
            Gaussian width along the grating direction
        sigma_y : float
            Gaussian width perpendicular to the grating direction
        n_stds : float
            Kernel half-size = n_stds * max(sigma_x, sigma_y)

        Returns
        -------
        real_kernel, imag_kernel : np.ndarray, shape (size, size)
        """
        max_sigma = max(sigma_x, sigma_y)
        half = int(np.ceil(n_stds * max_sigma))
        size = 2 * half + 1

        y_grid, x_grid = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x_grid * cos_t + y_grid * sin_t
        y_rot = -x_grid * sin_t + y_grid * cos_t

        gaussian = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        gaussian /= (gaussian.sum() + 1e-12)

        omega = 2.0 * np.pi * frequency
        real_kernel = gaussian * np.cos(omega * x_rot)
        imag_kernel = gaussian * np.sin(omega * x_rot)

        # DC correction
        real_kernel -= real_kernel.mean()

        return real_kernel, imag_kernel

    @staticmethod
    def bandwidth_to_sigma(frequency: float, bandwidth: float = 1.0) -> float:
        """
        Convert bandwidth (in octaves) to sigma.
        sigma = sqrt(ln2/2) * (2^B + 1) / (π * f * (2^B - 1))
        """
        if frequency < 1e-10:
            raise ValueError("Frequency must be positive.")
        return (np.sqrt(np.log(2) / 2.0) * (2 ** bandwidth + 1) /
                (np.pi * frequency * (2 ** bandwidth - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# 1D GABOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class GaborPipeline1D:
    """
    Full 1D Gabor transform pipeline.

    Steps
    -----
    1. Validate input signal and parameters
    2. Build Gabor filterbank (real + imaginary kernels)
    3. Convolve signal with each filter (complex convolution)
    4. Compute magnitude, phase, instantaneous frequency
    5. Compute energy and statistics per filter
    6. Visualize and save results
    """

    def __init__(self, params: GaborParams1D, config: PipelineConfig):
        self.params = params
        self.config = config
        self._kernels: List[Tuple[np.ndarray, np.ndarray]] = []
        os.makedirs(config.output_dir, exist_ok=True)

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_signal(self, sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float64).squeeze()
        if sig.ndim != 1:
            raise ValueError(f"1D pipeline expects 1D signal; got shape {sig.shape}")
        if len(sig) < 8:
            raise ValueError("Signal too short (minimum 8 samples).")
        if not np.all(np.isfinite(sig)):
            raise ValueError("Signal contains NaN or Inf values.")
        return sig

    def _validate_params(self, n: int):
        for f in self.params.frequencies:
            if not (0 < f <= 0.5):
                raise ValueError(f"Normalized frequency {f} must be in (0, 0.5].")
        for s in self.params.sigmas:
            if s <= 0:
                raise ValueError(f"Sigma {s} must be positive.")
            if 6 * s + 1 > n:
                warnings.warn(f"Sigma={s} yields kernel longer than signal (n={n}). "
                              "Consider reducing sigma.")

    # ── Filterbank construction ──────────────────────────────────────────────

    def _build_filterbank(self):
        self._kernels = []
        for f, s in zip(self.params.frequencies, self.params.sigmas):
            real_k, imag_k = GaborKernel1D.build(f, s, self.params.window_size)
            self._kernels.append((real_k, imag_k))
        if self.config.verbose:
            print(f"  [1D] Built {len(self._kernels)} Gabor filters.")

    # ── Convolution ──────────────────────────────────────────────────────────

    def _convolve(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(sig)
        nf = len(self._kernels)
        real_out = np.zeros((nf, n))
        imag_out = np.zeros((nf, n))

        for i, (rk, ik) in enumerate(self._kernels):
            # 'same' mode preserves signal length
            real_out[i] = signal.fftconvolve(sig, rk, mode='same')
            imag_out[i] = signal.fftconvolve(sig, ik, mode='same')

        return real_out, imag_out

    # ── Statistics ───────────────────────────────────────────────────────────

    def _compute_statistics(self, mag: np.ndarray, phase: np.ndarray,
                            real: np.ndarray) -> Dict[str, Any]:
        stats = {}
        for i in range(mag.shape[0]):
            stats[f'filter_{i}'] = {
                'energy': float(np.sum(mag[i] ** 2)),
                'mean_magnitude': float(mag[i].mean()),
                'std_magnitude': float(mag[i].std()),
                'max_magnitude': float(mag[i].max()),
                'peak_location': int(np.argmax(mag[i])),
                'phase_coherence': float(np.abs(np.mean(np.exp(1j * phase[i])))),
                'snr_db': float(
                    20 * np.log10(mag[i].max() / (mag[i].std() + 1e-12))
                ),
            }
        return stats

    # ── Main analysis ────────────────────────────────────────────────────────

    def analyze(self, sig: np.ndarray) -> AnalysisResult1D:
        t0 = time.time()

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("  1D GABOR TRANSFORM PIPELINE")
            print("=" * 60)

        sig = self._validate_signal(sig)
        self._validate_params(len(sig))
        self._build_filterbank()

        if self.config.verbose:
            print(f"  Signal length: {len(sig)}")
            print(f"  Frequencies: {self.params.frequencies}")
            print(f"  Sigmas:      {self.params.sigmas}")

        real_resp, imag_resp = self._convolve(sig)
        magnitude = np.sqrt(real_resp ** 2 + imag_resp ** 2)
        phase = np.arctan2(imag_resp, real_resp)
        energy = np.sum(magnitude ** 2, axis=1)

        stats = {}
        if self.config.compute_statistics:
            stats = self._compute_statistics(magnitude, phase, real_resp)

        result = AnalysisResult1D(
            signal_length=len(sig),
            n_filters=len(self._kernels),
            frequencies=np.array(self.params.frequencies),
            sigmas=np.array(self.params.sigmas),
            responses_real=real_resp,
            responses_imag=imag_resp,
            responses_magnitude=magnitude,
            responses_phase=phase,
            energy_per_filter=energy,
            statistics=stats,
            processing_time_s=time.time() - t0,
        )

        if self.config.verbose:
            print(f"  Processing time: {result.processing_time_s:.3f}s")

        if self.config.save_figures:
            self._plot_results(sig, result)
        if self.config.save_data:
            self._save_data(result)

        return result

    # ── Visualization ────────────────────────────────────────────────────────

    def _plot_results(self, sig: np.ndarray, result: AnalysisResult1D):
        nf = result.n_filters
        fig = plt.figure(figsize=(16, 4 + nf * 2.5))
        gs = gridspec.GridSpec(nf + 2, 3, figure=fig, hspace=0.45, wspace=0.35)

        t = np.arange(result.signal_length)

        # Original signal
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(t, sig, color='#1a1a2e', linewidth=0.8)
        ax0.set_title('Original Signal', fontsize=11, fontweight='bold')
        ax0.set_xlabel('Sample')
        ax0.set_ylabel('Amplitude')
        ax0.grid(True, alpha=0.3)

        # Scalogram (magnitude heatmap)
        ax1 = fig.add_subplot(gs[1, :])
        im = ax1.imshow(
            result.responses_magnitude,
            aspect='auto',
            origin='lower',
            extent=[0, result.signal_length, 0, nf],
            cmap='inferno',
        )
        ax1.set_yticks(np.arange(nf) + 0.5)
        ax1.set_yticklabels([f'f={f:.3f}' for f in result.frequencies], fontsize=7)
        ax1.set_title('Gabor Scalogram (Magnitude)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Filter')
        plt.colorbar(im, ax=ax1, label='Magnitude')

        # Per-filter plots
        for i in range(nf):
            row = i + 2
            f_label = f'f={result.frequencies[i]:.3f}, σ={result.sigmas[i]:.1f}'

            ax_real = fig.add_subplot(gs[row, 0])
            ax_real.plot(t, result.responses_real[i], color='#e63946', linewidth=0.7)
            ax_real.set_title(f'Real  [{f_label}]', fontsize=8)
            ax_real.grid(True, alpha=0.3)

            ax_mag = fig.add_subplot(gs[row, 1])
            ax_mag.plot(t, result.responses_magnitude[i], color='#2a9d8f', linewidth=0.7)
            ax_mag.set_title(f'Magnitude [{f_label}]', fontsize=8)
            ax_mag.grid(True, alpha=0.3)

            ax_phs = fig.add_subplot(gs[row, 2])
            ax_phs.plot(t, result.responses_phase[i], color='#f4a261', linewidth=0.7, alpha=0.85)
            ax_phs.set_title(f'Phase [{f_label}]', fontsize=8)
            ax_phs.set_ylabel('rad')
            ax_phs.grid(True, alpha=0.3)

        fig.suptitle('1D Gabor Transform Analysis', fontsize=14, fontweight='bold', y=1.001)
        out_path = Path(self.config.output_dir) / f'gabor_1d_analysis.{self.config.fig_format}'
        fig.savefig(out_path, dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close(fig)
        if self.config.verbose:
            print(f"  [1D] Figure saved: {out_path}")

    def _save_data(self, result: AnalysisResult1D):
        out = Path(self.config.output_dir)
        np.save(out / 'gabor_1d_magnitudes.npy', result.responses_magnitude)
        np.save(out / 'gabor_1d_phases.npy', result.responses_phase)
        np.save(out / 'gabor_1d_energy.npy', result.energy_per_filter)
        if result.statistics:
            with open(out / 'gabor_1d_statistics.json', 'w') as f:
                json.dump(result.statistics, f, indent=2)
        if self.config.verbose:
            print(f"  [1D] Data saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2D GABOR PIPELINE (single-band)
# ─────────────────────────────────────────────────────────────────────────────

class GaborPipeline2D:
    """
    Full 2D Gabor transform pipeline for grayscale images.

    Steps
    -----
    1. Validate input (arbitrary H×W)
    2. Build Gabor filterbank (n_freq × n_theta kernels)
    3. Convolve with each filter (FFT-based)
    4. Compute magnitude, phase maps
    5. Compute dominant orientation/frequency maps
    6. Energy analysis and statistics
    7. Visualize and save
    """

    def __init__(self, params: GaborParams2D, config: PipelineConfig):
        self.params = params
        self.config = config
        self._kernels: List[List[Tuple[np.ndarray, np.ndarray]]] = []
        os.makedirs(config.output_dir, exist_ok=True)

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_image(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 3:
            # Convert RGB to grayscale (luminance weights)
            if img.shape[2] == 3:
                img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            elif img.shape[2] == 1:
                img = img[..., 0]
            else:
                raise ValueError(f"Unexpected channel count: {img.shape[2]}")
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image; got shape {img.shape}")
        if not np.all(np.isfinite(img)):
            raise ValueError("Image contains NaN or Inf.")
        # Normalize to [0, 1]
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return img

    def _validate_params(self, H: int, W: int):
        for f in self.params.frequencies:
            if not (0 < f < 0.5):
                raise ValueError(f"Spatial frequency {f} must be in (0, 0.5).")
        for sx, sy in zip(self.params.sigmas_x, self.params.sigmas_y):
            if sx <= 0 or sy <= 0:
                raise ValueError("Sigma values must be positive.")

    # ── Filterbank ───────────────────────────────────────────────────────────

    def _build_filterbank(self):
        self._kernels = []
        nf, nt = len(self.params.frequencies), len(self.params.thetas)
        for fi, freq in enumerate(self.params.frequencies):
            row = []
            sx = self.params.sigmas_x[fi % len(self.params.sigmas_x)]
            sy = self.params.sigmas_y[fi % len(self.params.sigmas_y)]
            for theta in self.params.thetas:
                rk, ik = GaborKernel2D.build(
                    freq, theta, sx, sy, self.params.n_stds
                )
                row.append((rk, ik))
            self._kernels.append(row)
        if self.config.verbose:
            print(f"  [2D] Built {nf}×{nt} = {nf * nt} Gabor filters.")

    # ── Convolution (FFT-based for large kernels) ────────────────────────────

    def _convolve_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nf = len(self.params.frequencies)
        nt = len(self.params.thetas)
        H, W = img.shape
        mag = np.zeros((nf, nt, H, W), dtype=np.float32)
        phs = np.zeros((nf, nt, H, W), dtype=np.float32)

        for fi in range(nf):
            for ti in range(nt):
                rk, ik = self._kernels[fi][ti]
                real_r = signal.fftconvolve(img, rk, mode='same')
                imag_r = signal.fftconvolve(img, ik, mode='same')
                mag[fi, ti] = np.sqrt(real_r ** 2 + imag_r ** 2)
                phs[fi, ti] = np.arctan2(imag_r, real_r)

        return mag, phs

    # ── Feature maps ─────────────────────────────────────────────────────────

    def _compute_feature_maps(self, mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive energy, dominant orientation, and dominant frequency maps."""
        energy = np.sum(mag ** 2, axis=(0, 1))  # (H, W)

        # Dominant orientation index per pixel
        mag_per_theta = mag.sum(axis=0)  # (n_theta, H, W)
        dom_theta_idx = np.argmax(mag_per_theta, axis=0)  # (H, W)
        dom_theta = np.array(self.params.thetas)[dom_theta_idx]

        # Dominant frequency index per pixel
        mag_per_freq = mag.sum(axis=1)  # (n_freq, H, W)
        dom_freq_idx = np.argmax(mag_per_freq, axis=0)  # (H, W)
        dom_freq = np.array(self.params.frequencies)[dom_freq_idx]

        return energy, dom_theta, dom_freq

    # ── Statistics ───────────────────────────────────────────────────────────

    def _compute_statistics(self, mag: np.ndarray, energy: np.ndarray) -> Dict[str, Any]:
        stats = {
            'global_energy': float(energy.sum()),
            'mean_energy_per_pixel': float(energy.mean()),
            'per_filter': {}
        }
        for fi, freq in enumerate(self.params.frequencies):
            for ti, theta in enumerate(self.params.thetas):
                key = f'f{fi}_t{ti}'
                m = mag[fi, ti]
                stats['per_filter'][key] = {
                    'frequency': float(freq),
                    'theta_rad': float(theta),
                    'theta_deg': float(np.degrees(theta)),
                    'mean_magnitude': float(m.mean()),
                    'std_magnitude': float(m.std()),
                    'max_magnitude': float(m.max()),
                    'energy': float((m ** 2).sum()),
                }
        return stats

    # ── Main analysis ────────────────────────────────────────────────────────

    def analyze(self, img: np.ndarray, band_label: str = '') -> AnalysisResult2D:
        t0 = time.time()

        label = f"2D GABOR {'(' + band_label + ')' if band_label else ''}"
        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"  {label} PIPELINE")
            print('=' * 60)

        img = self._validate_image(img)
        H, W = img.shape
        self._validate_params(H, W)

        if self.config.verbose:
            print(f"  Image shape: {H}×{W}")

        self._build_filterbank()
        mag, phs = self._convolve_image(img)
        energy, dom_theta, dom_freq = self._compute_feature_maps(mag)

        stats = {}
        if self.config.compute_statistics:
            stats = self._compute_statistics(mag, energy)

        result = AnalysisResult2D(
            spatial_shape=(H, W),
            n_filters=len(self.params.frequencies) * len(self.params.thetas),
            n_frequencies=len(self.params.frequencies),
            n_orientations=len(self.params.thetas),
            frequencies=np.array(self.params.frequencies),
            thetas=np.array(self.params.thetas),
            responses_magnitude=mag,
            responses_phase=phs,
            energy_map=energy,
            dominant_orientation=dom_theta,
            dominant_frequency=dom_freq,
            statistics=stats,
            processing_time_s=time.time() - t0,
        )

        if self.config.verbose:
            print(f"  Processing time: {result.processing_time_s:.3f}s")

        if self.config.save_figures:
            self._plot_results(img, result, band_label)
        if self.config.save_data:
            self._save_data(result, band_label)

        return result

    # ── Visualization ────────────────────────────────────────────────────────

    def _plot_results(self, img: np.ndarray, result: AnalysisResult2D, label: str = ''):
        nf = result.n_frequencies
        nt = result.n_orientations

        fig = plt.figure(figsize=(5 * (nt + 1), 4 * (nf + 3)))
        total_rows = nf + 3

        gs = gridspec.GridSpec(total_rows, nt + 1, figure=fig,
                               hspace=0.4, wspace=0.3)

        # Original image
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(img, cmap='gray', aspect='auto')
        ax_orig.set_title(f'Input Image{" – " + label if label else ""}',
                          fontsize=12, fontweight='bold')
        ax_orig.axis('off')

        # Energy map
        ax_en = fig.add_subplot(gs[1, 0])
        im_en = ax_en.imshow(result.energy_map, cmap='hot', aspect='auto')
        ax_en.set_title('Integrated Energy', fontsize=9, fontweight='bold')
        ax_en.axis('off')
        plt.colorbar(im_en, ax=ax_en, fraction=0.046)

        # Dominant orientation
        ax_do = fig.add_subplot(gs[1, 1])
        im_do = ax_do.imshow(np.degrees(result.dominant_orientation),
                             cmap='hsv', aspect='auto', vmin=0, vmax=180)
        ax_do.set_title('Dominant Orientation (°)', fontsize=9, fontweight='bold')
        ax_do.axis('off')
        plt.colorbar(im_do, ax=ax_do, fraction=0.046)

        # Dominant frequency
        ax_df = fig.add_subplot(gs[1, 2] if nt >= 2 else gs[1, 1])
        im_df = ax_df.imshow(result.dominant_frequency, cmap='plasma', aspect='auto')
        ax_df.set_title('Dominant Frequency', fontsize=9, fontweight='bold')
        ax_df.axis('off')
        plt.colorbar(im_df, ax=ax_df, fraction=0.046)

        # Per-filter magnitude responses
        for fi in range(nf):
            for ti in range(nt):
                ax = fig.add_subplot(gs[fi + 2, ti])
                ax.imshow(result.responses_magnitude[fi, ti], cmap='viridis', aspect='auto')
                freq_str = f'{result.frequencies[fi]:.3f}'
                theta_str = f'{np.degrees(result.thetas[ti]):.0f}°'
                ax.set_title(f'f={freq_str}\nθ={theta_str}', fontsize=7)
                ax.axis('off')

        # Per-frequency energy bar chart
        ax_bar = fig.add_subplot(gs[-1, :])
        energy_by_freq = result.responses_magnitude.sum(axis=(1, 2, 3))
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, nf))
        ax_bar.bar(range(nf), energy_by_freq, color=colors)
        ax_bar.set_xticks(range(nf))
        ax_bar.set_xticklabels([f'{f:.3f}' for f in result.frequencies], fontsize=8)
        ax_bar.set_xlabel('Spatial Frequency')
        ax_bar.set_ylabel('Total Energy')
        ax_bar.set_title('Energy by Frequency', fontweight='bold')
        ax_bar.grid(True, alpha=0.3, axis='y')

        suffix = f'_{label}' if label else ''
        fig.suptitle(f'2D Gabor Transform Analysis{" – " + label if label else ""}',
                     fontsize=14, fontweight='bold')
        out = Path(self.config.output_dir) / f'gabor_2d{suffix}.{self.config.fig_format}'
        fig.savefig(out, dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close(fig)
        if self.config.verbose:
            print(f"  [2D] Figure saved: {out}")

    def _save_data(self, result: AnalysisResult2D, label: str = ''):
        suffix = f'_{label}' if label else ''
        out = Path(self.config.output_dir)
        np.save(out / f'gabor_2d{suffix}_magnitudes.npy', result.responses_magnitude)
        np.save(out / f'gabor_2d{suffix}_energy.npy', result.energy_map)
        np.save(out / f'gabor_2d{suffix}_dom_orientation.npy', result.dominant_orientation)
        if result.statistics:
            with open(out / f'gabor_2d{suffix}_statistics.json', 'w') as f:
                json.dump(result.statistics, f, indent=2)
        if self.config.verbose:
            print(f"  [2D] Data saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-BAND 2D GABOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class GaborPipelineMultiBand:
    """
    Gabor transform pipeline for multi-band 2D signals (H × W × L).

    Processing strategy
    -------------------
    - Each band is analyzed independently by GaborPipeline2D.
    - Inter-band correlation of energy maps is computed.
    - A fused energy map (mean across bands) is produced.
    - A cross-band orientation consistency map is derived.
    """

    def __init__(self, params: GaborParams2D, config: PipelineConfig):
        self.params = params
        self.config = config
        self._band_pipeline = GaborPipeline2D(params, config)
        os.makedirs(config.output_dir, exist_ok=True)

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_input(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            # Treat as single-band: add band axis
            data = data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(
                f"Multi-band pipeline expects H×W×L array; got shape {data.shape}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or Inf.")
        return data

    # ── Inter-band analysis ──────────────────────────────────────────────────

    def _inter_band_correlation(self, energy_maps: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation between energy maps of different bands.
        energy_maps: (L, H, W)
        Returns: (L, L) correlation matrix
        """
        L = energy_maps.shape[0]
        flat = energy_maps.reshape(L, -1)  # (L, H*W)
        # Normalize each band
        flat -= flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
        flat /= norms
        corr = flat @ flat.T  # (L, L)
        return corr

    def _orientation_consistency(self, dom_thetas: List[np.ndarray]) -> np.ndarray:
        """
        Circular mean resultant length of dominant orientations across bands.
        High value → bands agree on orientation at that pixel.
        """
        angles = np.stack(dom_thetas, axis=0)  # (L, H, W)
        # Orientation is π-periodic, double it for circular stats
        z = np.exp(2j * angles)
        resultant = np.abs(z.mean(axis=0))  # (H, W)
        return resultant

    # ── Main analysis ────────────────────────────────────────────────────────

    def analyze(self, data: np.ndarray) -> AnalysisResultMultiBand:
        t0 = time.time()

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("  MULTI-BAND 2D GABOR PIPELINE")
            print("=" * 60)

        data = self._validate_input(data)
        H, W, L = data.shape

        if self.config.verbose:
            print(f"  Data shape: {H}×{W}×{L} (H×W×L)")

        band_results: List[AnalysisResult2D] = []
        for band_idx in range(L):
            if self.config.verbose:
                print(f"\n  ── Band {band_idx + 1}/{L} ──")
            band_img = data[:, :, band_idx]
            result = self._band_pipeline.analyze(band_img, band_label=f'band{band_idx}')
            band_results.append(result)

        # Aggregate energy maps
        energy_maps = np.stack([r.energy_map for r in band_results], axis=0)  # (L, H, W)
        fused_energy = energy_maps.mean(axis=0)  # (H, W)

        inter_corr = self._inter_band_correlation(energy_maps)

        dom_thetas = [r.dominant_orientation for r in band_results]
        orientation_consistency = self._orientation_consistency(dom_thetas)

        stats = {
            'n_bands': L,
            'spatial_shape': [H, W],
            'inter_band_correlation_mean': float(
                (inter_corr.sum() - L) / max(1, L * (L - 1))
            ),
            'global_fused_energy': float(fused_energy.sum()),
            'mean_orientation_consistency': float(orientation_consistency.mean()),
        }

        result_mb = AnalysisResultMultiBand(
            spatial_shape=(H, W),
            n_bands=L,
            band_results=band_results,
            inter_band_correlation=inter_corr,
            fused_energy_map=fused_energy,
            statistics=stats,
            processing_time_s=time.time() - t0,
        )

        if self.config.verbose:
            print(f"\n  Total processing time: {result_mb.processing_time_s:.3f}s")

        if self.config.save_figures:
            self._plot_multiband(data, result_mb, orientation_consistency)
        if self.config.save_data:
            self._save_data(result_mb)

        return result_mb

    # ── Visualization ────────────────────────────────────────────────────────

    def _plot_multiband(self, data: np.ndarray, result: AnalysisResultMultiBand,
                        orient_consistency: np.ndarray):
        L = result.n_bands
        ncols = max(4, L)
        fig, axes = plt.subplots(4, ncols, figsize=(4 * ncols, 16))
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        # Row 0: band images
        for b in range(L):
            ax = axes[0, b]
            ax.imshow(data[:, :, b], cmap='gray', aspect='auto')
            ax.set_title(f'Band {b}', fontsize=9)
            ax.axis('off')
        for b in range(L, ncols):
            axes[0, b].axis('off')

        # Row 1: per-band energy maps
        for b in range(L):
            ax = axes[1, b]
            im = ax.imshow(result.band_results[b].energy_map, cmap='hot', aspect='auto')
            ax.set_title(f'Energy B{b}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        for b in range(L, ncols):
            axes[1, b].axis('off')

        # Row 2: fused energy, orientation consistency, inter-band correlation
        ax_fused = axes[2, 0]
        im_f = ax_fused.imshow(result.fused_energy_map, cmap='inferno', aspect='auto')
        ax_fused.set_title('Fused Energy (mean)', fontsize=9, fontweight='bold')
        ax_fused.axis('off')
        plt.colorbar(im_f, ax=ax_fused, fraction=0.046)

        ax_oc = axes[2, 1]
        im_oc = ax_oc.imshow(orient_consistency, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        ax_oc.set_title('Orientation Consistency', fontsize=9, fontweight='bold')
        ax_oc.axis('off')
        plt.colorbar(im_oc, ax=ax_oc, fraction=0.046)

        if result.inter_band_correlation is not None and result.n_bands > 1:
            ax_corr = axes[2, 2]
            im_corr = ax_corr.imshow(
                result.inter_band_correlation, cmap='RdBu_r',
                vmin=-1, vmax=1, aspect='equal'
            )
            ax_corr.set_title('Inter-Band Correlation', fontsize=9, fontweight='bold')
            ax_corr.set_xticks(range(L))
            ax_corr.set_yticks(range(L))
            plt.colorbar(im_corr, ax=ax_corr, fraction=0.046)

        for b in range(3, ncols):
            axes[2, b].axis('off')

        # Row 3: per-band dominant orientation
        for b in range(L):
            ax = axes[3, b]
            im_do = ax.imshow(
                np.degrees(result.band_results[b].dominant_orientation),
                cmap='hsv', aspect='auto', vmin=0, vmax=180
            )
            ax.set_title(f'Dom. Orient. B{b} (°)', fontsize=8)
            ax.axis('off')
            plt.colorbar(im_do, ax=ax, fraction=0.046)
        for b in range(L, ncols):
            axes[3, b].axis('off')

        fig.suptitle('Multi-Band 2D Gabor Transform Analysis', fontsize=14, fontweight='bold')
        out = Path(self.config.output_dir) / f'gabor_multiband.{self.config.fig_format}'
        fig.savefig(out, dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close(fig)
        if self.config.verbose:
            print(f"  [MultiB] Figure saved: {out}")

    def _save_data(self, result: AnalysisResultMultiBand):
        out = Path(self.config.output_dir)
        if result.fused_energy_map is not None:
            np.save(out / 'gabor_multiband_fused_energy.npy', result.fused_energy_map)
        if result.inter_band_correlation is not None:
            np.save(out / 'gabor_multiband_inter_corr.npy', result.inter_band_correlation)
        if result.statistics:
            with open(out / 'gabor_multiband_statistics.json', 'w') as f:
                json.dump(result.statistics, f, indent=2)
        if self.config.verbose:
            print(f"  [MultiB] Data saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY — UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

class GaborAnalysisPipeline:
    """
    High-level entry point. Dispatches to the correct sub-pipeline
    based on the input signal dimensionality.

    Usage
    -----
    >>> pipeline = GaborAnalysisPipeline(output_dir='./results')
    >>> result = pipeline.run(signal_or_image_or_cube)
    """

    # Default filterbank parameters
    DEFAULT_1D_FREQUENCIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    DEFAULT_1D_SIGMAS       = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0]

    DEFAULT_2D_FREQUENCIES  = [0.05, 0.10, 0.20, 0.35]
    DEFAULT_2D_THETAS       = [k * np.pi / 6 for k in range(6)]   # 0°–150° in 30° steps
    DEFAULT_2D_SIGMAS_X     = [8.0, 6.0, 4.0, 3.0]
    DEFAULT_2D_SIGMAS_Y     = [4.0, 3.0, 2.0, 1.5]

    def __init__(
        self,
        output_dir: str = './gabor_output',
        verbose: bool = True,
        fig_dpi: int = 150,
        params_1d: Optional[GaborParams1D] = None,
        params_2d: Optional[GaborParams2D] = None,
    ):
        self.config = PipelineConfig(
            output_dir=output_dir,
            verbose=verbose,
            fig_dpi=fig_dpi,
        )
        self.params_1d = params_1d or GaborParams1D(
            frequencies=self.DEFAULT_1D_FREQUENCIES,
            sigmas=self.DEFAULT_1D_SIGMAS,
        )
        self.params_2d = params_2d or GaborParams2D(
            frequencies=self.DEFAULT_2D_FREQUENCIES,
            thetas=self.DEFAULT_2D_THETAS,
            sigmas_x=self.DEFAULT_2D_SIGMAS_X,
            sigmas_y=self.DEFAULT_2D_SIGMAS_Y,
        )

    def run(
        self, data: np.ndarray
    ) -> Union[AnalysisResult1D, AnalysisResult2D, AnalysisResultMultiBand]:
        """
        Analyze a signal/image/cube with the Gabor transform.

        Parameters
        ----------
        data : np.ndarray
            - 1D array  → 1D pipeline
            - 2D array  → 2D pipeline (single grayscale image)
            - 3D array  → Multi-band 2D pipeline (H × W × L)

        Returns
        -------
        AnalysisResult1D | AnalysisResult2D | AnalysisResultMultiBand
        """
        data = np.asarray(data)

        if data.ndim == 1:
            pipeline = GaborPipeline1D(self.params_1d, self.config)
            return pipeline.analyze(data)

        elif data.ndim == 2:
            pipeline = GaborPipeline2D(self.params_2d, self.config)
            return pipeline.analyze(data)

        elif data.ndim == 3:
            pipeline = GaborPipelineMultiBand(self.params_2d, self.config)
            return pipeline.analyze(data)

        else:
            raise ValueError(
                f"Unsupported data dimensionality: {data.ndim}D. "
                "Expected 1D, 2D, or 3D (multi-band)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_signals():
    """Generate synthetic test signals for all three pipelines."""

    rng = np.random.default_rng(42)

    # ── 1D: multi-component chirp + noise ───────────────────────────────────
    n = 1024
    t = np.linspace(0, 1, n)
    sig_1d = (
        np.sin(2 * np.pi * 50 * t) * np.exp(-3 * t) +
        np.sin(2 * np.pi * 120 * t * (1 + 0.5 * t)) +
        0.3 * rng.standard_normal(n)
    )

    # ── 2D: synthetic texture (Gabor + noise + edge) ────────────────────────
    H, W = 128, 128
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    sig_2d = (
        np.cos(2 * np.pi * 0.08 * xx + 2 * np.pi * 0.04 * yy) *
        np.exp(-((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * 30 ** 2)) +
        np.cos(2 * np.pi * 0.15 * (xx * np.cos(np.pi / 4) + yy * np.sin(np.pi / 4))) +
        0.2 * rng.standard_normal((H, W))
    )

    # ── Multi-band: 4 bands = RGB + NIR (simulated) ─────────────────────────
    sig_mb = np.zeros((H, W, 4))
    for b, (fx, fy, noise_level) in enumerate([
        (0.08, 0.02, 0.15),
        (0.04, 0.10, 0.20),
        (0.12, 0.06, 0.10),
        (0.06, 0.14, 0.25),
    ]):
        sig_mb[:, :, b] = (
            np.cos(2 * np.pi * fx * xx + 2 * np.pi * fy * yy) +
            noise_level * rng.standard_normal((H, W))
        )

    return sig_1d, sig_2d, sig_mb


def run_demo(output_dir: str = './gabor_output'):
    """Run the full demo pipeline on synthetic signals."""
    print("\n" + "#" * 70)
    print("#  GABOR TRANSFORM ANALYSIS PIPELINE — DEMO RUN")
    print("#" * 70)

    sig_1d, sig_2d, sig_mb = generate_demo_signals()

    pipeline = GaborAnalysisPipeline(output_dir=output_dir, verbose=True)

    print("\n[1/3] Running 1D Gabor pipeline …")
    r1 = pipeline.run(sig_1d)
    print(f"      Energy per filter: {np.round(r1.energy_per_filter, 2)}")

    print("\n[2/3] Running 2D Gabor pipeline …")
    r2 = pipeline.run(sig_2d)
    print(f"      Global energy: {r2.statistics.get('global_energy', '–'):.2f}")

    print("\n[3/3] Running Multi-Band 2D Gabor pipeline …")
    r3 = pipeline.run(sig_mb)
    print(f"      Bands processed: {r3.n_bands}")
    print(f"      Mean inter-band correlation: "
          f"{r3.statistics.get('inter_band_correlation_mean', 0):.4f}")

    print(f"\n{'─' * 60}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"{'─' * 60}\n")

    return r1, r2, r3


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Rigorous Gabor Transform Analysis Pipeline'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run demo on synthetic signals')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to .npy file to analyze')
    parser.add_argument('--output-dir', type=str, default='./gabor_output',
                        help='Directory for output files')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI')
    args = parser.parse_args()

    if args.demo or args.input is None:
        run_demo(output_dir=args.output_dir)

    else:
        data = np.load(args.input)
        pipeline = GaborAnalysisPipeline(output_dir=args.output_dir, fig_dpi=args.dpi)
        result = pipeline.run(data)
        print(f"\nAnalysis complete. Type: {type(result).__name__}")
