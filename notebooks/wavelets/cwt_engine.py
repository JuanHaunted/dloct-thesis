"""
cwt_engine.py
-------------
Continuous Wavelet Transform implementations for:
  - 1-D signals
  - 2-D images (spatial CWT via ridge/row decomposition)
  - 3-D volumetric + time series
  - 2-D + N spectral bands (multi-band spatial signals)

All transforms are implemented via FFT-based convolution for efficiency.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CWTResult1D:
    """Result of a 1-D CWT."""
    scales: np.ndarray           # (n_scales,)
    frequencies: np.ndarray      # (n_scales,)  pseudo-frequencies
    coefficients: np.ndarray     # (n_scales, n_samples)  complex or real
    dt: float
    wavelet: str
    signal_length: int
    power: np.ndarray = field(init=False)

    def __post_init__(self):
        self.power = np.abs(self.coefficients) ** 2


@dataclass
class CWTResult2D:
    """Result of a 2-D CWT (per-row or 2-D isotropic)."""
    scales: np.ndarray           # (n_scales,)
    frequencies: np.ndarray      # (n_scales,)
    coefficients: np.ndarray     # (n_scales, H, W)
    wavelet: str
    mode: str                    # 'rows', 'cols', 'isotropic'
    power: np.ndarray = field(init=False)

    def __post_init__(self):
        self.power = np.abs(self.coefficients) ** 2


@dataclass
class CWTResult3DTime:
    """Result of a 3-D + time CWT: spatial volume per time frame."""
    scales: np.ndarray           # (n_scales,)
    frequencies: np.ndarray
    coefficients: np.ndarray     # (T, n_scales, D, H, W)
    wavelet: str
    dt: float
    power: np.ndarray = field(init=False)

    def __post_init__(self):
        self.power = np.abs(self.coefficients) ** 2


@dataclass
class CWTResultMultiBand:
    """Result of a 2-D + N-band CWT."""
    scales: np.ndarray           # (n_scales,)
    frequencies: np.ndarray
    band_names: List[str]
    coefficients: Dict[str, np.ndarray]   # band -> (n_scales, H, W)
    wavelet: str
    power: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.power = {b: np.abs(c)**2 for b, c in self.coefficients.items()}


# ---------------------------------------------------------------------------
# Scale / frequency helpers
# ---------------------------------------------------------------------------

def _build_scales(n_scales: int, min_scale: float, max_scale: float,
                  spacing: str = "log") -> np.ndarray:
    if spacing == "log":
        return np.geomspace(min_scale, max_scale, n_scales)
    return np.linspace(min_scale, max_scale, n_scales)


def _scale_to_freq(scales: np.ndarray, wavelet_name: str,
                   dt: float = 1.0) -> np.ndarray:
    """Approximate pseudo-frequency for each scale."""
    # Centre frequencies (approximate, wavelet-dependent)
    fc_map = {"morlet": 6.0 / (2 * np.pi),
              "mexican_hat": np.sqrt(2.5) / (2 * np.pi),
              "paul": 4.5 / (2 * np.pi),
              "dog": 1.0 / (2 * np.pi)}
    fc = fc_map.get(wavelet_name, 1.0 / (2 * np.pi))
    return fc / (scales * dt)


# ---------------------------------------------------------------------------
# Core FFT-based convolution
# ---------------------------------------------------------------------------

def _fft_cwt_1d(signal: np.ndarray, scales: np.ndarray,
                wavelet_freq_fn, dt: float = 1.0) -> np.ndarray:
    """
    Compute CWT of a 1-D signal via FFT convolution.
    Returns coefficients array of shape (n_scales, len(signal)).
    """
    N = len(signal)
    # FFT of signal (zero-padded to next power of 2 for speed)
    N_pad = int(2 ** np.ceil(np.log2(2 * N - 1)))
    sig_hat = np.fft.fft(signal, n=N_pad)
    omega = 2 * np.pi * np.fft.fftfreq(N_pad, d=dt)

    coeffs = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        # Scaled wavelet in frequency domain
        psi_hat = np.sqrt(2 * np.pi * s) * wavelet_freq_fn(s * omega)
        # Inverse FFT and trim
        conv = np.fft.ifft(sig_hat * np.conj(psi_hat))
        coeffs[i] = conv[:N]

    return coeffs


def _cwt_1d_direct(signal: np.ndarray, scales: np.ndarray,
                   wavelet_time_fn, dt: float = 1.0,
                   half_width: int = 512) -> np.ndarray:
    """
    Direct (time-domain) CWT for wavelets without analytic freq form.
    Slower but works for any wavelet.
    """
    N = len(signal)
    coeffs = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        # Build scaled, normalised wavelet on a grid ±half_width samples
        k = min(int(half_width * s), N * 3)
        t_grid = np.arange(-k, k + 1) * dt / s
        psi = wavelet_time_fn(t_grid) / np.sqrt(abs(s))
        # Convolve via FFT
        from scipy.signal import fftconvolve
        conv = fftconvolve(signal.astype(complex), psi[::-1], mode='same')
        coeffs[i] = conv * dt
    return coeffs


# ---------------------------------------------------------------------------
# 1-D CWT
# ---------------------------------------------------------------------------

def cwt_1d(signal: np.ndarray,
           wavelet: str = "morlet",
           n_scales: int = 64,
           min_scale: float = 1.0,
           max_scale: Optional[float] = None,
           dt: float = 1.0,
           scale_spacing: str = "log") -> CWTResult1D:
    """
    Compute the 1-D Continuous Wavelet Transform.

    Parameters
    ----------
    signal    : 1-D array of shape (N,)
    wavelet   : 'morlet' | 'mexican_hat' | 'paul' | 'dog'
    n_scales  : number of scales
    min_scale : smallest scale (in samples)
    max_scale : largest scale  (default: N/4 samples)
    dt        : sampling interval
    scale_spacing : 'log' | 'linear'

    Returns
    -------
    CWTResult1D
    """
    from wavelets import get_wavelet
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if max_scale is None:
        max_scale = N / 4.0

    scales = _build_scales(n_scales, min_scale, max_scale, scale_spacing)
    winfo = get_wavelet(wavelet)

    if winfo["freq"] is not None:
        coeffs = _fft_cwt_1d(signal, scales, winfo["freq"], dt)
    else:
        coeffs = _cwt_1d_direct(signal, scales, winfo["time"], dt)

    if not winfo["complex"]:
        coeffs = coeffs.real

    freqs = _scale_to_freq(scales, wavelet, dt)
    return CWTResult1D(scales=scales, frequencies=freqs,
                       coefficients=coeffs, dt=dt,
                       wavelet=wavelet, signal_length=N)


# ---------------------------------------------------------------------------
# 2-D CWT
# ---------------------------------------------------------------------------

def cwt_2d(image: np.ndarray,
           wavelet: str = "morlet",
           n_scales: int = 32,
           min_scale: float = 1.0,
           max_scale: Optional[float] = None,
           mode: str = "isotropic",
           dt: float = 1.0,
           scale_spacing: str = "log") -> CWTResult2D:
    """
    Compute the 2-D CWT of an image.

    Parameters
    ----------
    image  : 2-D array of shape (H, W)
    mode   : 'rows'      — apply 1-D CWT along each row (H independent CWTs)
             'cols'      — apply 1-D CWT along each column
             'isotropic' — separable 2-D CWT (rows × cols averaged in power)

    Returns
    -------
    CWTResult2D  with coefficients shape (n_scales, H, W)
    """
    from wavelets import get_wavelet
    image = np.asarray(image, dtype=float)
    H, W = image.shape
    if max_scale is None:
        max_scale = min(H, W) / 4.0

    scales = _build_scales(n_scales, min_scale, max_scale, scale_spacing)
    winfo = get_wavelet(wavelet)
    freqs = _scale_to_freq(scales, wavelet, dt)

    coeffs = np.zeros((len(scales), H, W), dtype=complex)

    if mode in ("rows", "isotropic"):
        # CWT along each row
        for r in range(H):
            if winfo["freq"] is not None:
                row_coeffs = _fft_cwt_1d(image[r], scales, winfo["freq"], dt)
            else:
                row_coeffs = _cwt_1d_direct(image[r], scales, winfo["time"], dt)
            coeffs[:, r, :] += row_coeffs

    if mode in ("cols", "isotropic"):
        col_coeffs_acc = np.zeros_like(coeffs)
        for c in range(W):
            if winfo["freq"] is not None:
                col_coeffs = _fft_cwt_1d(image[:, c], scales, winfo["freq"], dt)
            else:
                col_coeffs = _cwt_1d_direct(image[:, c], scales, winfo["time"], dt)
            col_coeffs_acc[:, :, c] += col_coeffs

        if mode == "isotropic":
            # geometric mean of row and column contributions
            coeffs = np.sqrt(np.abs(coeffs) * np.abs(col_coeffs_acc)) * np.exp(
                1j * (np.angle(coeffs) + np.angle(col_coeffs_acc)) / 2)
        else:
            coeffs = col_coeffs_acc

    if not winfo["complex"]:
        coeffs = coeffs.real

    return CWTResult2D(scales=scales, frequencies=freqs,
                       coefficients=coeffs, wavelet=wavelet, mode=mode)


# ---------------------------------------------------------------------------
# 3-D + Time CWT
# ---------------------------------------------------------------------------

def cwt_3d_time(volume_series: np.ndarray,
                wavelet: str = "morlet",
                n_scales: int = 24,
                min_scale: float = 1.0,
                max_scale: Optional[float] = None,
                dt: float = 1.0,
                scale_spacing: str = "log") -> CWTResult3DTime:
    """
    Compute CWT of a 4-D array (T, D, H, W) — volume time series.

    For each voxel (d, h, w), a 1-D CWT is computed along the time axis.
    This is the most memory-intensive mode; reduce n_scales if needed.

    Parameters
    ----------
    volume_series : array (T, D, H, W)

    Returns
    -------
    CWTResult3DTime with coefficients shape (T, n_scales, D, H, W)
    Note: axes reordered so you index [t, scale, d, h, w].
    """
    from wavelets import get_wavelet
    volume_series = np.asarray(volume_series, dtype=float)
    T, D, H, W = volume_series.shape

    if max_scale is None:
        max_scale = T / 4.0

    scales = _build_scales(n_scales, min_scale, max_scale, scale_spacing)
    winfo = get_wavelet(wavelet)
    freqs = _scale_to_freq(scales, wavelet, dt)

    # Output shape: (n_scales, T, D, H, W) → transpose later
    coeffs = np.zeros((len(scales), T, D, H, W), dtype=complex)

    for d in range(D):
        for h in range(H):
            for w in range(W):
                ts = volume_series[:, d, h, w]
                if winfo["freq"] is not None:
                    c = _fft_cwt_1d(ts, scales, winfo["freq"], dt)
                else:
                    c = _cwt_1d_direct(ts, scales, winfo["time"], dt)
                coeffs[:, :, d, h, w] = c  # (n_scales, T)

    # Rearrange to (T, n_scales, D, H, W)
    coeffs = np.transpose(coeffs, (1, 0, 2, 3, 4))

    if not winfo["complex"]:
        coeffs = coeffs.real

    return CWTResult3DTime(scales=scales, frequencies=freqs,
                           coefficients=coeffs, wavelet=wavelet, dt=dt)


# ---------------------------------------------------------------------------
# 2-D + N spectral bands CWT
# ---------------------------------------------------------------------------

def cwt_multiband(band_images: Dict[str, np.ndarray],
                  wavelet: str = "morlet",
                  n_scales: int = 32,
                  min_scale: float = 1.0,
                  max_scale: Optional[float] = None,
                  mode: str = "isotropic",
                  dt: float = 1.0,
                  scale_spacing: str = "log") -> CWTResultMultiBand:
    """
    Compute the 2-D CWT for each spectral band independently.

    Parameters
    ----------
    band_images : dict mapping band_name -> 2-D array (H, W)
    mode        : passed to cwt_2d for each band

    Returns
    -------
    CWTResultMultiBand
    """
    if not band_images:
        raise ValueError("band_images must contain at least one band.")

    # Validate shapes
    shapes = {name: img.shape for name, img in band_images.items()}
    first_shape = next(iter(shapes.values()))
    for name, sh in shapes.items():
        if sh != first_shape:
            raise ValueError(f"Band '{name}' has shape {sh} but expected {first_shape}.")

    H, W = first_shape
    if max_scale is None:
        max_scale = min(H, W) / 4.0

    scales_ref = _build_scales(n_scales, min_scale, max_scale, scale_spacing)
    from wavelets import get_wavelet
    winfo = get_wavelet(wavelet)
    freqs = _scale_to_freq(scales_ref, wavelet, dt)

    band_coeffs = {}
    for band_name, img in band_images.items():
        result = cwt_2d(img, wavelet=wavelet, n_scales=n_scales,
                        min_scale=min_scale, max_scale=max_scale,
                        mode=mode, dt=dt, scale_spacing=scale_spacing)
        band_coeffs[band_name] = result.coefficients

    return CWTResultMultiBand(
        scales=scales_ref,
        frequencies=freqs,
        band_names=list(band_images.keys()),
        coefficients=band_coeffs,
        wavelet=wavelet,
    )
