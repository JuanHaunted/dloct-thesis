"""
wavelets.py
-----------
Wavelet definitions for the CWT pipeline.
All wavelets are implemented as analytic functions returning complex-valued
mother wavelets in both time and frequency domains.
"""

import numpy as np
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Mother wavelet library
# ---------------------------------------------------------------------------

def morlet(t: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """Morlet (Gabor) wavelet — complex, good TF localisation."""
    norm = np.pi**-0.25
    return norm * np.exp(1j * omega0 * t) * np.exp(-0.5 * t**2)


def morlet_freq(omega: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """Morlet wavelet in frequency domain (for fast CWT)."""
    norm = np.pi**-0.25 * np.sqrt(2 * np.pi)
    heaviside = (omega > 0).astype(float)
    return norm * np.exp(-0.5 * (omega - omega0)**2) * heaviside


def mexican_hat(t: np.ndarray) -> np.ndarray:
    """Mexican hat (Ricker) — 2nd derivative of Gaussian, real-valued."""
    c = 2.0 / (np.sqrt(3) * np.pi**0.25)
    return c * (1 - t**2) * np.exp(-0.5 * t**2)


def mexican_hat_freq(omega: np.ndarray) -> np.ndarray:
    """Mexican hat in frequency domain."""
    c = -np.sqrt(8.0 / 3.0) * np.pi**0.25
    return c * omega**2 * np.exp(-0.5 * omega**2)


def paul(t: np.ndarray, m: int = 4) -> np.ndarray:
    """Paul wavelet of order m (complex)."""
    from scipy.special import factorial
    norm = (2**m * factorial(m)) / np.sqrt(np.pi * factorial(2 * m))
    return norm * (1 - 1j * t) ** (-(m + 1))


def paul_freq(omega: np.ndarray, m: int = 4) -> np.ndarray:
    """Paul wavelet in frequency domain."""
    from scipy.special import factorial
    norm = (2**m) / np.sqrt(m * factorial(2 * m))
    heaviside = (omega > 0).astype(float)
    return norm * omega**m * np.exp(-omega) * heaviside


def dog(t: np.ndarray, m: int = 2) -> np.ndarray:
    """Derivative of Gaussian (DOG) wavelet of order m."""
    # (-1)^(m+1) * d^m/dt^m [Gaussian]
    from scipy.special import factorial
    from scipy.stats import norm as snorm
    g = snorm.pdf(t)
    # numerical derivative via finite differences (simple, accurate enough)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    result = g.copy()
    for _ in range(m):
        result = np.gradient(result, dt)
    sign = (-1)**(m + 1)
    norm = 1.0 / np.sqrt(factorial(m))
    return sign * norm * result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WAVELET_REGISTRY = {
    "morlet":       {"time": morlet,        "freq": morlet_freq,        "complex": True},
    "mexican_hat":  {"time": mexican_hat,   "freq": mexican_hat_freq,   "complex": False},
    "paul":         {"time": paul,          "freq": paul_freq,          "complex": True},
    "dog":          {"time": dog,           "freq": None,               "complex": False},
}


def get_wavelet(name: str) -> dict:
    name = name.lower().replace("-", "_")
    if name not in WAVELET_REGISTRY:
        raise ValueError(f"Unknown wavelet '{name}'. Available: {list(WAVELET_REGISTRY)}")
    return WAVELET_REGISTRY[name]
