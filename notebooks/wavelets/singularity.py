"""
singularity.py
--------------
Multifractal / singularity-spectrum analysis via the Wavelet Transform
Modulus Maxima (WTMM) method and the direct Hölder exponent estimation.

Theory
------
For a signal f(t), the local Hölder (Lipschitz) exponent α(t₀) measures the
local regularity.  The singularity spectrum D(α) is the Hausdorff dimension of
the set of points with Hölder exponent α.

WTMM pipeline:
  1. Compute CWT coefficients W(s, t).
  2. Find modulus maxima (ridges) at each scale s.
  3. Compute partition function Z(q, s) = Σ_{maxima} |W(s, t_max)|^q
  4. Estimate τ(q) from Z(q,s) ~ s^(τ(q)+1) via log-log regression.
  5. Legendre transform: α(q) = dτ/dq,  D(α) = qα − τ(q).
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.signal import argrelmax


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SingularitySpectrum:
    """Full multifractal analysis result."""
    alpha: np.ndarray        # Hölder exponents
    D_alpha: np.ndarray      # Singularity spectrum D(α)
    q_values: np.ndarray     # q-moments used
    tau: np.ndarray          # Scaling exponents τ(q)
    hurst: float             # Hurst exponent ≈ α at q=2 (monofractal proxy)
    alpha_max: float         # α where D(α) is maximum (most probable exponent)
    width: float             # Spectrum width (max_alpha − min_alpha)
    log_scales: np.ndarray   # log2 of scales used
    log_partition: np.ndarray  # (n_q, n_scales) log2 Z(q,s)


@dataclass
class HolderField:
    """Pointwise Hölder exponents for a 1-D signal."""
    t: np.ndarray          # time axis
    alpha: np.ndarray      # local Hölder exponent at each point
    ridge_scales: np.ndarray  # dominant scale at each point


# ---------------------------------------------------------------------------
# Modulus maxima detection
# ---------------------------------------------------------------------------

def _find_modulus_maxima(coeffs: np.ndarray,
                          order: int = 3) -> List[np.ndarray]:
    """
    Find local maxima of |W(s, t)| along the time axis for each scale.

    Returns
    -------
    maxima_per_scale : list of length n_scales, each element is an array of
                       indices (along time) that are local maxima.
    """
    n_scales, N = coeffs.shape
    power = np.abs(coeffs)
    maxima_per_scale = []
    for i in range(n_scales):
        row = power[i]
        idx = argrelmax(row, order=order)[0]
        maxima_per_scale.append(idx)
    return maxima_per_scale


# ---------------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------------

def _partition_function(coeffs: np.ndarray,
                         scales: np.ndarray,
                         q_values: np.ndarray,
                         maxima_only: bool = True,
                         order: int = 3) -> np.ndarray:
    """
    Compute partition function Z(q, s).

    If maxima_only=True, sum only over WTMM (Wavelet Transform Modulus Maxima).
    Otherwise, sum over all coefficients.

    Returns
    -------
    Z : array (n_q, n_scales)
    """
    n_q = len(q_values)
    n_scales = len(scales)
    power = np.abs(coeffs)  # (n_scales, N)

    if maxima_only:
        maxima = _find_modulus_maxima(coeffs, order=order)

    Z = np.zeros((n_q, n_scales))
    for si in range(n_scales):
        if maxima_only and len(maxima[si]) > 0:
            vals = power[si, maxima[si]]
        elif not maxima_only:
            vals = power[si]
        else:
            vals = np.array([1e-30])  # no maxima found → small placeholder

        vals = np.where(vals < 1e-30, 1e-30, vals)
        for qi, q in enumerate(q_values):
            Z[qi, si] = np.sum(vals ** q)

    return Z


# ---------------------------------------------------------------------------
# Scaling exponent τ(q)
# ---------------------------------------------------------------------------

def _estimate_tau(Z: np.ndarray,
                  scales: np.ndarray,
                  q_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate τ(q) via log-log linear regression of Z(q,s) vs s.

    Z(q,s) ~ s^(τ(q)+1)  →  log Z = (τ(q)+1) log s + const

    Returns
    -------
    tau      : array (n_q,)
    log_part : log2 of partition function, shape (n_q, n_scales)
    """
    log_s = np.log2(scales)          # (n_scales,)
    log_Z = np.log2(np.maximum(Z, 1e-30))  # (n_q, n_scales)

    tau = np.zeros(len(q_values))
    for qi in range(len(q_values)):
        # Robust linear regression: τ(q)+1 is the slope
        slope, intercept = np.polyfit(log_s, log_Z[qi], 1)
        tau[qi] = slope - 1.0

    return tau, log_Z


# ---------------------------------------------------------------------------
# Legendre transform
# ---------------------------------------------------------------------------

def _legendre_transform(q_values: np.ndarray,
                         tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute singularity spectrum via discrete Legendre transform.

    α(q) = dτ/dq   (numerical gradient)
    D(α) = qα − τ(q)
    """
    alpha = np.gradient(tau, q_values)
    D_alpha = q_values * alpha - tau
    return alpha, D_alpha


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def singularity_spectrum(cwt_result,
                          q_min: float = -5.0,
                          q_max: float = 5.0,
                          n_q: int = 41,
                          scale_range: Optional[Tuple[int, int]] = None,
                          maxima_only: bool = True) -> SingularitySpectrum:
    """
    Compute the multifractal singularity spectrum D(α) from a CWT result.

    Parameters
    ----------
    cwt_result   : CWTResult1D (or any object with .coefficients and .scales)
    q_min, q_max : range of q moments
    n_q          : number of q values
    scale_range  : (i_min, i_max) scale indices to use (None = all)
    maxima_only  : if True, use WTMM method; else sum all coefficients

    Returns
    -------
    SingularitySpectrum
    """
    coeffs = cwt_result.coefficients  # (n_scales, N)
    scales = cwt_result.scales

    # Restrict to scale range
    if scale_range is not None:
        i0, i1 = scale_range
        coeffs = coeffs[i0:i1]
        scales = scales[i0:i1]

    q_values = np.linspace(q_min, q_max, n_q)
    # Exclude q ≈ 0 (would give division issues in tau)
    q_values = q_values[np.abs(q_values) > 0.05]

    Z = _partition_function(coeffs, scales, q_values, maxima_only=maxima_only)
    tau, log_Z = _estimate_tau(Z, scales, q_values)
    alpha, D_alpha = _legendre_transform(q_values, tau)

    # Keep only physically meaningful part (D ≥ 0 approximately)
    valid = D_alpha > -0.5
    alpha_v = alpha[valid]
    D_v = D_alpha[valid]

    # Hurst exponent: at q=2 → α(2) ≈ H (monofractal proxy)
    idx2 = np.argmin(np.abs(q_values - 2.0))
    hurst = float(alpha[idx2])

    # Most probable exponent (peak of D(α))
    if len(D_v) > 0:
        alpha_max = float(alpha_v[np.argmax(D_v)])
    else:
        alpha_max = float(np.nanmean(alpha))

    width = float(np.nanmax(alpha_v) - np.nanmin(alpha_v)) if len(alpha_v) > 1 else 0.0

    return SingularitySpectrum(
        alpha=alpha_v,
        D_alpha=D_v,
        q_values=q_values,
        tau=tau,
        hurst=hurst,
        alpha_max=alpha_max,
        width=width,
        log_scales=np.log2(scales),
        log_partition=log_Z,
    )


def holder_exponents_1d(cwt_result,
                         scale_range: Optional[Tuple[int, int]] = None) -> HolderField:
    """
    Compute pointwise local Hölder exponents from the CWT cone of influence.

    For each time point t, the dominant scale s*(t) is identified as the
    scale of maximum wavelet power.  The local exponent is estimated as:
        α(t) ≈ log|W(s*, t)| / log(s*)

    Parameters
    ----------
    cwt_result : CWTResult1D

    Returns
    -------
    HolderField
    """
    coeffs = cwt_result.coefficients
    scales = cwt_result.scales

    if scale_range is not None:
        i0, i1 = scale_range
        coeffs = coeffs[i0:i1]
        scales = scales[i0:i1]

    power = np.abs(coeffs)  # (n_scales, N)
    N = power.shape[1]

    # Dominant scale index per time point
    dom_scale_idx = np.argmax(power, axis=0)  # (N,)
    dom_scale = scales[dom_scale_idx]         # (N,)

    # Power at dominant scale
    dom_power = power[dom_scale_idx, np.arange(N)]
    dom_power = np.where(dom_power < 1e-30, 1e-30, dom_power)

    # Local Hölder exponent
    alpha = np.log2(dom_power + 1e-30) / np.log2(dom_scale + 1e-10)

    t = np.arange(N) * cwt_result.dt
    return HolderField(t=t, alpha=alpha, ridge_scales=dom_scale)


def multiscale_entropy(cwt_result, q: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scale-resolved Rényi entropy.

    H_q(s) = 1/(1-q) * log2( Σ_t p(s,t)^q )
    where p(s,t) = |W(s,t)|^2 / Σ_t |W(s,t)|^2

    Returns
    -------
    scales    : (n_scales,)
    entropy   : (n_scales,)
    """
    power = np.abs(cwt_result.coefficients) ** 2
    scales = cwt_result.scales

    entropy = np.zeros(len(scales))
    for i, s in enumerate(scales):
        p = power[i] / (power[i].sum() + 1e-30)
        p = np.where(p < 1e-30, 1e-30, p)
        if abs(q - 1.0) < 1e-6:
            # Shannon entropy
            entropy[i] = -np.sum(p * np.log2(p))
        else:
            entropy[i] = np.log2(np.sum(p**q)) / (1 - q)

    return scales, entropy
