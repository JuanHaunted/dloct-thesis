"""
pipeline.py
-----------
High-level Pipeline class that wraps cwt_engine, singularity, and visualizer
into a single unified API.

Usage
-----
    from pipeline import WaveletPipeline
    import numpy as np

    pipe = WaveletPipeline(wavelet="morlet", n_scales=64)

    # 1-D
    t = np.linspace(0, 10, 1024)
    signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*20*t)
    result = pipe.analyze_1d(signal, dt=t[1]-t[0])

    # 2-D
    image = np.random.randn(128, 128)
    result2 = pipe.analyze_2d(image)

    # Multi-band
    bands = {"red": np.random.randn(64,64), "green": np.random.randn(64,64)}
    result_mb = pipe.analyze_multiband(bands)
"""

import sys
import os
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt

# Allow running from any directory
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from cwt_engine import cwt_1d, cwt_2d, cwt_3d_time, cwt_multiband
from singularity import (singularity_spectrum, holder_exponents_1d,
                          multiscale_entropy, SingularitySpectrum, HolderField)
from visualizer import (plot_cwt_1d, plot_cwt_2d, plot_cwt_3d_time,
                         plot_cwt_multiband, plot_singularity_spectrum,
                         plot_multiscale_entropy)


class WaveletPipeline:
    """
    Unified pipeline for multi-dimensional CWT analysis and singularity spectra.

    Parameters
    ----------
    wavelet       : 'morlet' | 'mexican_hat' | 'paul' | 'dog'
    n_scales      : number of scales
    min_scale     : smallest scale (samples)
    max_scale     : largest scale  (None → auto)
    scale_spacing : 'log' | 'linear'
    output_dir    : directory for saving figures (created if missing)
    """

    def __init__(self,
                 wavelet: str = "morlet",
                 n_scales: int = 64,
                 min_scale: float = 1.0,
                 max_scale: Optional[float] = None,
                 scale_spacing: str = "log",
                 output_dir: str = "./cwt_output"):
        self.wavelet = wavelet
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_spacing = scale_spacing
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1-D analysis
    # ------------------------------------------------------------------

    def analyze_1d(self,
                   signal: np.ndarray,
                   dt: float = 1.0,
                   compute_spectrum: bool = True,
                   compute_holder: bool = True,
                   compute_entropy: bool = True,
                   q_min: float = -5.0,
                   q_max: float = 5.0,
                   n_q: int = 41,
                   save_figures: bool = True,
                   tag: str = "1d") -> dict:
        """
        Full 1-D wavelet analysis pipeline.

        Returns
        -------
        dict with keys: 'cwt', 'spectrum', 'holder', 'entropy'
        """
        print(f"[1D] Computing CWT ({self.wavelet}, {self.n_scales} scales)…")
        result = cwt_1d(signal, wavelet=self.wavelet, n_scales=self.n_scales,
                         min_scale=self.min_scale, max_scale=self.max_scale,
                         dt=dt, scale_spacing=self.scale_spacing)

        out = {"cwt": result}

        if save_figures:
            fig = plot_cwt_1d(result, signal=signal, title=f"CWT 1-D [{self.wavelet}]",
                               save_path=os.path.join(self.output_dir, f"{tag}_scalogram.png"))
            plt.close(fig)

        if compute_spectrum:
            print("[1D] Computing singularity spectrum…")
            ss = singularity_spectrum(result, q_min=q_min, q_max=q_max, n_q=n_q)
            out["spectrum"] = ss
            holder = holder_exponents_1d(result) if compute_holder else None
            out["holder"] = holder
            if save_figures:
                fig = plot_singularity_spectrum(
                    ss, holder=holder,
                    title=f"Singularity Spectrum [{self.wavelet}]",
                    save_path=os.path.join(self.output_dir, f"{tag}_singularity.png"))
                plt.close(fig)

        if compute_entropy:
            print("[1D] Computing multiscale entropy…")
            scales, entropy = multiscale_entropy(result)
            out["entropy"] = (scales, entropy)
            if save_figures:
                fig = plot_multiscale_entropy(
                    scales, entropy,
                    save_path=os.path.join(self.output_dir, f"{tag}_entropy.png"))
                plt.close(fig)

        print(f"[1D] Done. Hurst exponent ≈ {out.get('spectrum', None) and out['spectrum'].hurst:.4f}")
        return out

    # ------------------------------------------------------------------
    # 2-D analysis
    # ------------------------------------------------------------------

    def analyze_2d(self,
                   image: np.ndarray,
                   mode: str = "isotropic",
                   scale_indices: Optional[List[int]] = None,
                   save_figures: bool = True,
                   tag: str = "2d") -> dict:
        """
        Full 2-D wavelet analysis pipeline.

        Parameters
        ----------
        image : 2-D array (H, W)
        mode  : 'rows' | 'cols' | 'isotropic'
        """
        print(f"[2D] Computing CWT ({self.wavelet}, mode={mode})…")
        result = cwt_2d(image, wavelet=self.wavelet, n_scales=self.n_scales,
                         min_scale=self.min_scale, max_scale=self.max_scale,
                         mode=mode, scale_spacing=self.scale_spacing)

        if save_figures:
            fig = plot_cwt_2d(result, image=image,
                               scale_indices=scale_indices,
                               title=f"CWT 2-D [{self.wavelet}, {mode}]",
                               save_path=os.path.join(self.output_dir, f"{tag}_cwt2d.png"))
            plt.close(fig)

        print("[2D] Done.")
        return {"cwt": result}

    # ------------------------------------------------------------------
    # 3-D + Time analysis
    # ------------------------------------------------------------------

    def analyze_3d_time(self,
                         volume_series: np.ndarray,
                         dt: float = 1.0,
                         time_indices: Optional[List[int]] = None,
                         scale_index: int = 0,
                         save_figures: bool = True,
                         tag: str = "3dt") -> dict:
        """
        Full 3-D + time wavelet analysis pipeline.

        Parameters
        ----------
        volume_series : 4-D array (T, D, H, W)
        """
        T, D, H, W = volume_series.shape
        print(f"[3D+T] Computing CWT for volume ({T}×{D}×{H}×{W})…")
        result = cwt_3d_time(volume_series, wavelet=self.wavelet,
                              n_scales=self.n_scales,
                              min_scale=self.min_scale,
                              max_scale=self.max_scale,
                              dt=dt, scale_spacing=self.scale_spacing)

        if save_figures:
            fig = plot_cwt_3d_time(result, time_indices=time_indices,
                                    scale_index=scale_index,
                                    title=f"CWT 3-D+Time [{self.wavelet}]",
                                    save_path=os.path.join(self.output_dir,
                                                            f"{tag}_cwt3dt.png"))
            plt.close(fig)

        print("[3D+T] Done.")
        return {"cwt": result}

    # ------------------------------------------------------------------
    # 2-D + N-band analysis
    # ------------------------------------------------------------------

    def analyze_multiband(self,
                           band_images: Dict[str, np.ndarray],
                           mode: str = "isotropic",
                           scale_index: int = 0,
                           save_figures: bool = True,
                           tag: str = "mb") -> dict:
        """
        Full multi-band 2-D wavelet analysis pipeline.

        Parameters
        ----------
        band_images : dict band_name -> (H, W) array
        """
        bands = list(band_images.keys())
        print(f"[MultiB] Computing CWT for {len(bands)} bands: {bands}…")
        result = cwt_multiband(band_images, wavelet=self.wavelet,
                                n_scales=self.n_scales,
                                min_scale=self.min_scale,
                                max_scale=self.max_scale,
                                mode=mode, scale_spacing=self.scale_spacing)

        if save_figures:
            fig = plot_cwt_multiband(result, scale_index=scale_index,
                                      band_images=band_images,
                                      title=f"Multi-band CWT [{self.wavelet}]",
                                      save_path=os.path.join(self.output_dir,
                                                              f"{tag}_multiband.png"))
            plt.close(fig)

        print("[MultiB] Done.")
        return {"cwt": result}

    # ------------------------------------------------------------------
    # Convenience: full demo
    # ------------------------------------------------------------------

    def run_demo(self):
        """Generate synthetic data for all modes and run the full pipeline."""
        rng = np.random.default_rng(42)
        print("=" * 60)
        print("WAVELET PIPELINE DEMO")
        print("=" * 60)

        # -------- 1-D: multi-component signal + fractal noise --------
        dt = 1.0 / 512
        t = np.arange(0, 4.0, dt)
        # Chirp + harmonic + fractional Brownian motion proxy
        signal = (np.sin(2 * np.pi * 10 * t) +
                  0.5 * np.sin(2 * np.pi * (50 + 30 * t) * t) +
                  np.cumsum(rng.normal(0, 0.1, len(t))))
        self.analyze_1d(signal, dt=dt, tag="demo_1d")

        # -------- 2-D: synthetic textured image --------
        H, W = 128, 128
        xx, yy = np.meshgrid(np.linspace(0, 4*np.pi, W), np.linspace(0, 4*np.pi, H))
        image = np.sin(xx) * np.cos(yy) + 0.5 * rng.standard_normal((H, W))
        self.analyze_2d(image, mode="isotropic", tag="demo_2d")

        # -------- 3-D + Time --------
        T, D, H2, W2 = 8, 4, 16, 16
        vol = np.sin(np.linspace(0, 4*np.pi, T))[:, None, None, None] + \
              0.1 * rng.standard_normal((T, D, H2, W2))
        self.analyze_3d_time(vol, dt=dt, tag="demo_3dt")

        # -------- Multi-band --------
        H3, W3 = 64, 64
        xx2, yy2 = np.meshgrid(np.linspace(0, 2*np.pi, W3),
                                np.linspace(0, 2*np.pi, H3))
        bands = {
            "red":   np.sin(xx2) + 0.2 * rng.standard_normal((H3, W3)),
            "green": np.cos(yy2) + 0.2 * rng.standard_normal((H3, W3)),
            "blue":  np.sin(xx2 + yy2) + 0.2 * rng.standard_normal((H3, W3)),
            "nir":   np.cos(xx2 - yy2) + 0.2 * rng.standard_normal((H3, W3)),
        }
        self.analyze_multiband(bands, tag="demo_mb")

        print("=" * 60)
        print(f"All outputs saved to: {os.path.abspath(self.output_dir)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wavelet Analysis Pipeline")
    parser.add_argument("--wavelet", default="morlet",
                        choices=["morlet", "mexican_hat", "paul", "dog"])
    parser.add_argument("--n-scales", type=int, default=48)
    parser.add_argument("--output-dir", default="./cwt_output")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    pipe = WaveletPipeline(wavelet=args.wavelet,
                            n_scales=args.n_scales,
                            output_dir=args.output_dir)
    if args.demo:
        pipe.run_demo()
    else:
        print("Specify --demo to run the built-in demo, or import WaveletPipeline in your code.")
