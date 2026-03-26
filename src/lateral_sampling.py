"""
Lateral Sampling Analysis for OCT Tomograms

This module provides tools for analyzing and controlling lateral sampling in OCT
tomograms using Mean Power Spectrum (MPS) analysis with Gaussian fitting.

Mathematical Foundation:
------------------------
The lateral PSF of an OCT system follows a Gaussian profile due to the Gaussian
beam used in scanning. The power spectrum of a Gaussian is also Gaussian:

    P(f) = A * exp(-f^2 / (2*sigma_f^2)) + offset

where sigma_f is the spectral bandwidth (standard deviation in frequency domain).

The Half-Width at Half-Maximum (HWHM) relates to sigma by:
    HWHM = sigma * sqrt(2 * ln(2)) ≈ 1.177 * sigma

Nyquist Criterion:
------------------
For normalized frequency (f_norm ∈ [-0.5, 0.5]):
- A signal is properly sampled when its bandwidth fits within the Nyquist limit
- When subsampling by factor k, the normalized bandwidth appears to increase by k
- Aliasing occurs when the apparent bandwidth exceeds 0.5

Author: DLOCT Thesis Project
Date: 2024
"""

import numpy as np
from numpy.fft import fft, fftshift, ifftshift, fft2
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum


class LateralAxis(Enum):
    """Enumeration for lateral axes in OCT tomograms."""
    FAST = 0  # x-axis (fast scanning direction)
    SLOW = 1  # y-axis (slow scanning direction)


@dataclass
class GaussianFitResult:
    """Result of Gaussian fitting to power spectrum."""
    amplitude: float      # Peak amplitude A
    sigma: float          # Standard deviation in normalized frequency
    offset: float         # DC offset
    hwhm: float           # Half-width at half-maximum
    r_squared: float      # Goodness of fit (R^2)
    fit_curve: np.ndarray # Fitted Gaussian curve
    freq_axis: np.ndarray # Frequency axis used for fitting

    @property
    def fwhm(self) -> float:
        """Full-width at half-maximum."""
        return 2 * self.hwhm

    @property
    def is_undersampled(self) -> bool:
        """Check if the signal exceeds Nyquist (HWHM > 0.5 in normalized freq)."""
        return self.hwhm > 0.5


@dataclass
class ThresholdBandwidthResult:
    """Result of threshold-based bandwidth measurement.

    Unlike Gaussian HWHM which captures the central peak, this metric
    measures the actual spectral extent by finding where the spectrum
    drops below a given threshold (e.g., 1% of peak).
    """
    threshold: float      # Threshold as fraction of peak (e.g., 0.01 for 1%)
    half_width: float     # Half-width at the threshold level
    full_width: float     # Full width at the threshold level
    freq_axis: np.ndarray # Frequency axis
    mps: np.ndarray       # Power spectrum used

    @property
    def is_undersampled(self) -> bool:
        """Check if spectral content exceeds Nyquist at this threshold."""
        return self.half_width > 0.5


def compute_spectral_halfwidth(
    freq: np.ndarray,
    mps: np.ndarray,
    threshold: float = 0.01
) -> ThresholdBandwidthResult:
    """
    Compute the half-width of the spectrum at a given threshold level.

    Unlike Gaussian fitting which captures the central peak, this metric
    directly measures where the spectrum drops below a threshold fraction
    of the peak value. This captures the actual spectral extent including
    the tails.

    Parameters
    ----------
    freq : np.ndarray
        Normalized frequency axis (should span [-0.5, 0.5] approximately)
    mps : np.ndarray
        Mean power spectrum (normalized so peak ~ 1)
    threshold : float
        Threshold as fraction of peak (default 0.01 = 1%)

    Returns
    -------
    ThresholdBandwidthResult
        Dataclass containing half-width and related metrics

    Notes
    -----
    For a Nyquist-sampled signal, the half-width at 1% should be ~0.5.
    After decimation by factor k, the half-width scales approximately as:
        HW_k ≈ min(k × HW_0, 0.5)

    The half-width is capped at 0.5 because that's the maximum observable
    frequency in normalized coordinates. When the true half-width exceeds 0.5,
    aliasing causes energy to wrap around.

    Examples
    --------
    >>> freq, mps = compute_mps_1d(enface, axis=0)
    >>> result = compute_spectral_halfwidth(freq, mps, threshold=0.01)
    >>> print(f"Half-width at 1%: {result.half_width:.3f}")
    """
    # Ensure MPS is normalized
    mps_norm = mps / np.max(mps)

    # Find the peak location (should be near f=0)
    peak_idx = np.argmax(mps_norm)

    # Search outward from peak to find where spectrum drops below threshold
    # We search in both directions and take the average

    n = len(freq)

    # Search right from peak
    right_idx = peak_idx
    for i in range(peak_idx, n):
        if mps_norm[i] < threshold:
            right_idx = i
            break
    else:
        right_idx = n - 1  # Spectrum doesn't drop below threshold

    # Search left from peak
    left_idx = peak_idx
    for i in range(peak_idx, -1, -1):
        if mps_norm[i] < threshold:
            left_idx = i
            break
    else:
        left_idx = 0  # Spectrum doesn't drop below threshold

    # Get the frequencies at the threshold crossings
    # Use linear interpolation for more accuracy
    if right_idx < n - 1 and right_idx > peak_idx:
        # Interpolate to find exact crossing point
        f1, f2 = freq[right_idx - 1], freq[right_idx]
        m1, m2 = mps_norm[right_idx - 1], mps_norm[right_idx]
        if m1 != m2:
            freq_right = f1 + (threshold - m1) * (f2 - f1) / (m2 - m1)
        else:
            freq_right = freq[right_idx]
    else:
        freq_right = freq[right_idx]

    if left_idx > 0 and left_idx < peak_idx:
        f1, f2 = freq[left_idx], freq[left_idx + 1]
        m1, m2 = mps_norm[left_idx], mps_norm[left_idx + 1]
        if m1 != m2:
            freq_left = f1 + (threshold - m1) * (f2 - f1) / (m2 - m1)
        else:
            freq_left = freq[left_idx]
    else:
        freq_left = freq[left_idx]

    # Compute half-width (average of left and right extents from center)
    # The center is at f=0, not necessarily at the peak
    half_width_right = abs(freq_right)
    half_width_left = abs(freq_left)
    half_width = (half_width_right + half_width_left) / 2

    full_width = freq_right - freq_left

    return ThresholdBandwidthResult(
        threshold=threshold,
        half_width=half_width,
        full_width=full_width,
        freq_axis=freq,
        mps=mps_norm
    )


@dataclass
class SubsamplingAnalysis:
    """Analysis result for a specific subsampling factor."""
    factor: float                    # Subsampling factor (can be fractional)
    original_hwhm: float             # HWHM of original tomogram
    apparent_hwhm: float             # HWHM after subsampling+interpolation
    theoretical_hwhm: float          # Expected HWHM = factor * original_hwhm
    is_undersampled: bool            # Whether apparent_hwhm > 0.5
    fit_result: GaussianFitResult    # Full fit details


def gaussian(f: np.ndarray, A: float, sigma: float, offset: float) -> np.ndarray:
    """
    Gaussian function for power spectrum fitting.

    Parameters
    ----------
    f : np.ndarray
        Frequency values (centered at 0)
    A : float
        Peak amplitude
    sigma : float
        Standard deviation (spectral bandwidth)
    offset : float
        DC offset (noise floor)

    Returns
    -------
    np.ndarray
        Gaussian values at each frequency

    Notes
    -----
    The function is: g(f) = A * exp(-f^2 / (2*sigma^2)) + offset
    """
    return A * np.exp(-f**2 / (2 * sigma**2)) + offset


def compute_mps_1d(
    data: np.ndarray,
    axis: int,
    average_axis: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D Mean Power Spectrum along specified axis.

    Parameters
    ----------
    data : np.ndarray
        2D array (e.g., en-face plane or B-scan)
    axis : int
        Axis along which to compute FFT (0 or 1)
    average_axis : int, optional
        Axis along which to average the power spectrum.
        If None, averages along the other axis.

    Returns
    -------
    freq : np.ndarray
        Normalized frequency axis [-0.5, 0.5)
    mps : np.ndarray
        Mean power spectrum (normalized to peak = 1)

    Notes
    -----
    The power spectrum is computed as |FFT|^2 and then averaged
    across the orthogonal axis. The result is normalized so the
    maximum value is 1.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    if average_axis is None:
        average_axis = 1 - axis  # The other axis

    n_samples = data.shape[axis]

    # Compute FFT with proper centering
    # fftshift before FFT centers the spatial data (optional but consistent)
    # fftshift after FFT centers the frequency spectrum
    ft = fftshift(fft(ifftshift(data, axes=axis), axis=axis), axes=axis)

    # Power spectrum
    power = np.abs(ft)**2

    # Average along specified axis
    mps = np.mean(power, axis=average_axis)

    # Normalize to peak = 1
    mps = mps / np.max(mps)

    # Normalized frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(n_samples))

    return freq, mps


def compute_mps_2d(
    data: np.ndarray,
    average_dim: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D Mean Power Spectrum and average along one dimension.

    Parameters
    ----------
    data : np.ndarray
        2D array (e.g., en-face plane)
    average_dim : int
        Dimension to average over after 2D FFT (0 or 1)

    Returns
    -------
    freq : np.ndarray
        Normalized frequency axis for the non-averaged dimension
    mps : np.ndarray
        Mean power spectrum along the non-averaged dimension
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    # 2D FFT with centering
    ft = fftshift(fft2(ifftshift(data)))

    # Power spectrum
    power = np.abs(ft)**2

    # Average along specified dimension
    mps = np.mean(power, axis=average_dim)

    # Normalize
    mps = mps / np.max(mps)

    # Frequency axis for the other dimension
    n_samples = data.shape[1 - average_dim]
    freq = np.fft.fftshift(np.fft.fftfreq(n_samples))

    return freq, mps


def fit_gaussian_to_mps(
    freq: np.ndarray,
    mps: np.ndarray,
    p0: Optional[Tuple[float, float, float]] = None
) -> GaussianFitResult:
    """
    Fit a Gaussian to the Mean Power Spectrum.

    Parameters
    ----------
    freq : np.ndarray
        Normalized frequency axis (should be centered at 0)
    mps : np.ndarray
        Mean power spectrum values (should be normalized to peak ~ 1)
    p0 : tuple, optional
        Initial guess for (amplitude, sigma, offset).
        If None, estimated from data.

    Returns
    -------
    GaussianFitResult
        Dataclass containing fit parameters and quality metrics

    Notes
    -----
    The fitting uses scipy.optimize.curve_fit with the Levenberg-Marquardt
    algorithm. The HWHM is computed from sigma as: HWHM = sigma * sqrt(2*ln(2))
    """
    if p0 is None:
        # Estimate initial parameters
        A0 = np.max(mps) - np.min(mps)
        offset0 = np.min(mps)
        # Estimate sigma from the half-power points
        half_max = (np.max(mps) + np.min(mps)) / 2
        above_half = np.abs(mps) > half_max
        if np.any(above_half):
            sigma0 = np.sum(above_half) / len(mps) * 0.5  # Rough estimate
        else:
            sigma0 = 0.1
        p0 = (A0, sigma0, offset0)

    # Bounds to ensure physically meaningful parameters
    bounds = (
        [0, 0.001, 0],           # Lower bounds: A >= 0, sigma > 0, offset >= 0
        [np.inf, 1.0, np.inf]    # Upper bounds: sigma <= 1 (normalized freq)
    )

    try:
        popt, pcov = curve_fit(
            gaussian, freq, mps, p0=p0,
            bounds=bounds,
            maxfev=10000
        )
    except RuntimeError as e:
        raise ValueError(f"Gaussian fitting failed: {e}")

    A, sigma, offset = popt

    # Compute R^2 (coefficient of determination)
    fit_curve = gaussian(freq, *popt)
    residuals = mps - fit_curve
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mps - np.mean(mps))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # HWHM from sigma
    hwhm = sigma * np.sqrt(2 * np.log(2))

    return GaussianFitResult(
        amplitude=A,
        sigma=sigma,
        offset=offset,
        hwhm=hwhm,
        r_squared=r_squared,
        fit_curve=fit_curve,
        freq_axis=freq
    )


def subsample_lateral(
    data: np.ndarray,
    factor: int,
    axis: int,
    interpolate: bool = True
) -> np.ndarray:
    """
    Subsample data along a lateral axis with optional interpolation.

    This function implements TRUE decimation (keeping every k-th sample)
    WITHOUT anti-aliasing filtering, which causes aliasing when the signal
    bandwidth exceeds the new Nyquist frequency.

    Parameters
    ----------
    data : np.ndarray
        Input array (2D or 3D)
    factor : int
        Subsampling factor (keep every factor-th sample)
    axis : int
        Axis along which to subsample
    interpolate : bool
        If True, interpolate back to original size using FFT-based
        zero-padding (sinc interpolation)

    Returns
    -------
    np.ndarray
        Subsampled (and optionally interpolated) array

    Notes
    -----
    Integer subsampling keeps every factor-th sample:
        subsampled[i] = original[i * factor]

    This introduces aliasing when the signal bandwidth exceeds the new
    Nyquist frequency (f_N' = f_N / factor).

    The interpolation uses FFT zero-padding which is equivalent to
    ideal sinc interpolation. This restores the original sample count
    but the aliased content remains, causing the apparent bandwidth
    to increase by factor k in normalized frequency coordinates.
    """
    if factor < 1:
        raise ValueError(f"Subsampling factor must be >= 1, got {factor}")

    if factor == 1:
        return data.copy()

    original_shape = data.shape
    original_n = original_shape[axis]

    # Create slice for subsampling (TRUE decimation - no filtering)
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, factor)
    subsampled = data[tuple(slices)]

    if not interpolate:
        return subsampled

    # FFT-based interpolation (sinc interpolation via zero-padding)
    # This preserves the frequency content including aliased components
    return _fft_interpolate(subsampled, original_n, axis)


def _linear_interpolate_1d(data: np.ndarray, target_n: int, axis: int) -> np.ndarray:
    """
    Interpolate data using linear interpolation along one axis.

    Parameters
    ----------
    data : np.ndarray
        Input array
    target_n : int
        Target size along axis
    axis : int
        Axis to interpolate along

    Returns
    -------
    np.ndarray
        Interpolated array with size target_n along axis
    """
    current_n = data.shape[axis]

    if current_n == target_n:
        return data.copy()

    # Create interpolation coordinates
    x_old = np.linspace(0, 1, current_n)
    x_new = np.linspace(0, 1, target_n)

    # Move axis to front
    data_moved = np.moveaxis(data, axis, 0)
    old_shape = data_moved.shape
    data_flat = data_moved.reshape(current_n, -1)

    # Interpolate each column
    if np.iscomplexobj(data):
        real_interp = np.zeros((target_n, data_flat.shape[1]), dtype=np.float64)
        imag_interp = np.zeros((target_n, data_flat.shape[1]), dtype=np.float64)
        for i in range(data_flat.shape[1]):
            real_interp[:, i] = np.interp(x_new, x_old, data_flat[:, i].real)
            imag_interp[:, i] = np.interp(x_new, x_old, data_flat[:, i].imag)
        result_flat = real_interp + 1j * imag_interp
    else:
        result_flat = np.zeros((target_n, data_flat.shape[1]), dtype=data.dtype)
        for i in range(data_flat.shape[1]):
            result_flat[:, i] = np.interp(x_new, x_old, data_flat[:, i])

    # Reshape and move axis back
    new_shape = (target_n,) + old_shape[1:]
    result = result_flat.reshape(new_shape)
    result = np.moveaxis(result, 0, axis)

    return result


def subsample_lateral_fractional(
    data: np.ndarray,
    factor: float,
    axis: int
) -> np.ndarray:
    """
    Subsample data with fractional factor using decimation + interpolation.

    This implements decimation (keeping every k-th sample approximately)
    followed by linear interpolation back to original size.

    Parameters
    ----------
    data : np.ndarray
        Input array (2D or 3D)
    factor : float
        Subsampling factor (can be non-integer, e.g., 1.5)
    axis : int
        Axis along which to subsample

    Returns
    -------
    np.ndarray
        Subsampled and interpolated array with original shape

    Notes
    -----
    **Important for MPS Analysis:**
    To correctly measure the HWHM increase due to subsampling, you should
    measure the MPS on the DECIMATED signal (before interpolation), not on
    the interpolated signal. Use `subsample_lateral(..., interpolate=False)`
    for MPS measurements.

    This function is primarily for creating training pairs where both
    input and target have the same dimensions.

    For factor=2 and N=512:
    - Intermediate size: 256 samples (decimated)
    - Final size: 512 samples (interpolated)
    - MPS should be measured on the 256-sample version
    """
    if factor < 1:
        raise ValueError(f"Subsampling factor must be >= 1, got {factor}")

    if factor == 1.0:
        return data.copy()

    original_size = data.shape[axis]

    # For integer factors, use exact decimation
    if factor == int(factor):
        k = int(factor)
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(None, None, k)
        decimated = data[tuple(slices)]
    else:
        # For fractional factors, compute target intermediate size
        intermediate_size = int(np.floor(original_size / factor))
        intermediate_size = max(intermediate_size, 2)
        # Use linear interpolation to get to intermediate size
        decimated = _linear_interpolate_1d(data, intermediate_size, axis)

    # Interpolate back to original size
    return _linear_interpolate_1d(decimated, original_size, axis)


def analyze_subsampling(
    plane: np.ndarray,
    axis: int,
    factors: np.ndarray,
    average_axis: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze how the MPS half-width changes with subsampling.

    **Important:** The MPS is measured on the DECIMATED signal (before
    interpolation) to correctly capture the bandwidth stretching effect
    in normalized frequency coordinates.

    Parameters
    ----------
    plane : np.ndarray
        2D en-face plane or B-scan
    axis : int
        Lateral axis to analyze (0 or 1)
    factors : np.ndarray
        Array of subsampling factors to test (should be integers for accuracy)
    average_axis : int, optional
        Axis to average MPS over. If None, uses the other axis.

    Returns
    -------
    dict
        Dictionary containing:
        - 'factors': subsampling factors tested
        - 'hwhm': HWHM values for each factor
        - 'r_squared': fit quality for each factor
        - 'original_fit': GaussianFitResult for original data
        - 'critical_factor': factor at which HWHM = 0.5
        - 'analyses': list of SubsamplingAnalysis objects

    Notes
    -----
    The theory predicts: HWHM_apparent = factor × HWHM_original

    This is because decimation reduces the Nyquist frequency by factor k,
    so the same physical bandwidth now spans a larger fraction of the
    normalized frequency range [-0.5, 0.5].

    The critical factor is where HWHM reaches 0.5 (Nyquist limit):
        k_critical = 0.5 / HWHM_original
    """
    if average_axis is None:
        average_axis = 1 - axis

    # Analyze original (factor = 1)
    freq_orig, mps_orig = compute_mps_1d(plane, axis, average_axis)
    fit_orig = fit_gaussian_to_mps(freq_orig, mps_orig)

    hwhm_values = []
    r_squared_values = []
    analyses = []

    for factor in factors:
        if factor == 1.0:
            hwhm_values.append(fit_orig.hwhm)
            r_squared_values.append(fit_orig.r_squared)
            analyses.append(SubsamplingAnalysis(
                factor=factor,
                original_hwhm=fit_orig.hwhm,
                apparent_hwhm=fit_orig.hwhm,
                theoretical_hwhm=fit_orig.hwhm,
                is_undersampled=fit_orig.is_undersampled,
                fit_result=fit_orig
            ))
            continue

        # Decimate WITHOUT interpolation - keep every k-th sample
        k = int(np.round(factor))
        slices = [slice(None)] * plane.ndim
        slices[axis] = slice(None, None, k)
        decimated = plane[tuple(slices)]

        # Compute MPS on decimated signal
        # The frequency axis spans [-0.5, 0.5] in normalized coordinates
        # The same physical bandwidth now appears wider
        freq, mps = compute_mps_1d(decimated, axis, average_axis)

        try:
            fit = fit_gaussian_to_mps(freq, mps)
            hwhm_values.append(fit.hwhm)
            r_squared_values.append(fit.r_squared)

            analyses.append(SubsamplingAnalysis(
                factor=factor,
                original_hwhm=fit_orig.hwhm,
                apparent_hwhm=fit.hwhm,
                theoretical_hwhm=factor * fit_orig.hwhm,
                is_undersampled=fit.hwhm > 0.5,
                fit_result=fit
            ))
        except ValueError:
            hwhm_values.append(np.nan)
            r_squared_values.append(np.nan)

    # Find critical factor using theoretical relationship
    # k_critical = 0.5 / HWHM_original
    critical_factor = 0.5 / fit_orig.hwhm if fit_orig.hwhm > 0 else np.inf

    return {
        'factors': np.array(factors),
        'hwhm': np.array(hwhm_values),
        'r_squared': np.array(r_squared_values),
        'original_fit': fit_orig,
        'critical_factor': critical_factor,
        'analyses': analyses
    }


def determine_subsampling_factor(
    tomogram: np.ndarray,
    target_hwhm: float = 0.6,
    axis: int = 1,
    depth_slices: Optional[np.ndarray] = None,
    n_samples: int = 10
) -> Dict[str, Any]:
    """
    Determine the subsampling factor needed to achieve target undersampling.

    Parameters
    ----------
    tomogram : np.ndarray
        3D OCT tomogram with shape (depth, x, y) or (depth, x, y, 2)
    target_hwhm : float
        Target HWHM in normalized frequency (> 0.5 for undersampling)
    axis : int
        Lateral axis to subsample (1 for fast, 2 for slow in 3D array)
    depth_slices : np.ndarray, optional
        Which depth indices to analyze. If None, uses central slices.
    n_samples : int
        Number of depth slices to average over

    Returns
    -------
    dict
        Dictionary containing:
        - 'recommended_factor': subsampling factor to achieve target
        - 'original_hwhm': average HWHM of original tomogram
        - 'depth_analysis': per-depth analysis results

    Notes
    -----
    The recommended factor is: target_hwhm / original_hwhm

    For training a super-resolution model, target_hwhm > 0.5 ensures
    the subsampled data is genuinely undersampled (aliased).
    """
    # Handle complex representation
    if tomogram.ndim == 4 and tomogram.shape[-1] == 2:
        tom = tomogram[..., 0] + 1j * tomogram[..., 1]
    elif tomogram.ndim == 4:
        # Polarimetric data - use first channel
        tom = tomogram[..., 0]
    else:
        tom = tomogram

    if depth_slices is None:
        # Use central portion of the tomogram
        center = tom.shape[0] // 2
        depth_slices = np.linspace(
            center - tom.shape[0] // 4,
            center + tom.shape[0] // 4,
            n_samples
        ).astype(int)

    hwhm_per_depth = []

    for z in depth_slices:
        if z < 0 or z >= tom.shape[0]:
            continue

        plane = tom[z, :, :]

        # Analyze in the specified axis direction
        # For 2D plane, axis 0 is x (fast), axis 1 is y (slow)
        analysis_axis = axis - 1 if axis > 0 else 0

        freq, mps = compute_mps_1d(plane, analysis_axis)

        try:
            fit = fit_gaussian_to_mps(freq, mps)
            hwhm_per_depth.append(fit.hwhm)
        except ValueError:
            continue

    if not hwhm_per_depth:
        raise ValueError("Could not fit Gaussian to any depth slice")

    original_hwhm = np.mean(hwhm_per_depth)
    std_hwhm = np.std(hwhm_per_depth)

    # Calculate recommended factor
    recommended_factor = target_hwhm / original_hwhm

    return {
        'recommended_factor': recommended_factor,
        'original_hwhm': original_hwhm,
        'hwhm_std': std_hwhm,
        'target_hwhm': target_hwhm,
        'depth_slices': depth_slices,
        'hwhm_per_depth': np.array(hwhm_per_depth),
        'is_oversampled': original_hwhm < 0.5
    }


def create_training_pair(
    tomogram: np.ndarray,
    subsampling_factor: float,
    axis: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a training pair (subsampled, original) for super-resolution.

    Parameters
    ----------
    tomogram : np.ndarray
        3D or 4D OCT tomogram
    subsampling_factor : float
        Factor by which to subsample
    axis : int
        Lateral axis to subsample (1 or 2 in 3D indexing)

    Returns
    -------
    subsampled : np.ndarray
        Subsampled and interpolated tomogram (same shape as input)
    original : np.ndarray
        Copy of original tomogram

    Notes
    -----
    The subsampled tomogram has the same shape as the original but
    with reduced lateral resolution (information loss).
    """
    original = tomogram.copy()

    # Apply subsampling to each depth slice
    if tomogram.ndim == 4:
        # Real/Imag or polarimetric
        subsampled = np.zeros_like(tomogram)
        for z in range(tomogram.shape[0]):
            for c in range(tomogram.shape[3]):
                subsampled[z, :, :, c] = subsample_lateral_fractional(
                    tomogram[z, :, :, c], subsampling_factor, axis - 1
                )
    elif tomogram.ndim == 3:
        subsampled = np.zeros_like(tomogram)
        for z in range(tomogram.shape[0]):
            subsampled[z] = subsample_lateral_fractional(
                tomogram[z], subsampling_factor, axis - 1
            )
    else:
        raise ValueError(f"Expected 3D or 4D tomogram, got shape {tomogram.shape}")

    return subsampled, original
