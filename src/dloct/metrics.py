"""
Amplitude and phase fidelity metrics (docs/literature_review.md §5).

Inputs are complex tensors (B, H, W), lateral axis last, in normalized units (the per-volume
99.9th-percentile amplitude is 1). Phase metrics are restricted to a tissue mask
``|x| > 10^(tissue_db/20)``; speckle phase in the background is noise and is not scored.
"""

import math

import torch
import torch.nn.functional as F

from .losses import lateral_phasor

_EPS = 1e-12


def tissue_mask(x, tissue_db: float = -30.0):
    return x.abs() > 10 ** (tissue_db / 20)


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _masked_mean(v, w):
    return (v * w).sum() / (w.sum() + _EPS)


def _box(v, k):
    return F.avg_pool2d(v.unsqueeze(1), k, stride=1, padding=k // 2,
                        count_include_pad=False).squeeze(1)


def to_db_image(x, dyn_range_db: float = 50.0):
    """20·log10|x| clipped to [−dyn_range, 0] dB and scaled to [0, 1]."""
    db = 20 * torch.log10(x.abs() + _EPS)
    return ((db + dyn_range_db) / dyn_range_db).clamp(0, 1)


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(-2, -1))
    return (10 * torch.log10(1.0 / (mse + _EPS))).mean()


def ssim(a, b, window: int = 11, sigma: float = 1.5):
    """Gaussian-window SSIM on [0, 1] images (B, H, W)."""
    g = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    a, b = a.unsqueeze(1), b.unsqueeze(1)
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    var_a = F.conv2d(a * a, k) - mu_a ** 2
    var_b = F.conv2d(b * b, k) - mu_b ** 2
    cov = F.conv2d(a * b, k) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean()


@torch.no_grad()
def compute_metrics(x_hat, x, tissue_db: float = -30.0, local_window: int = 5):
    """Returns a dict of floats. ``x_hat`` and ``x`` are complex (B, H, W)."""
    x_hat, x = x_hat.to(torch.complex64), x.to(torch.complex64)
    mask = tissue_mask(x, tissue_db).float()
    out = {}

    # Amplitude, dB domain.
    a_hat, a = to_db_image(x_hat), to_db_image(x)
    out["psnr_db"] = psnr(a_hat, a).item()
    out["ssim_db"] = ssim(a_hat, a).item()

    # Complex field.
    out["nrmse"] = ((x_hat - x).abs().pow(2).sum() / (x.abs().pow(2).sum() + _EPS)).sqrt().item()
    rho = (x_hat * x.conj()).sum() / torch.sqrt(x_hat.abs().pow(2).sum() * x.abs().pow(2).sum() + _EPS)
    out["rho_global"] = rho.abs().item()

    cross = x_hat * x.conj()
    num = torch.complex(_box(cross.real, local_window), _box(cross.imag, local_window))
    den = torch.sqrt(_box(x_hat.abs() ** 2, local_window) * _box(x.abs() ** 2, local_window) + _EPS)
    out["rho_local"] = _masked_mean(num.abs() / den, mask).item()

    # Absolute phase (tissue only).
    dphi = _wrap(torch.angle(x_hat) - torch.angle(x)).abs()
    out["phase_err_rad"] = _masked_mean(dphi, mask).item()
    out["phase_err_w_rad"] = _masked_mean(dphi, mask * x.abs() ** 2).item()

    # Inter-A-line phase difference (Kasai 3x3), the quantity Doppler/OCE use.
    d_hat, d = lateral_phasor(x_hat), lateral_phasor(x)
    m_d = mask[..., 1:] * mask[..., :-1]
    dd = _wrap(torch.angle(d_hat) - torch.angle(d)).abs()
    out["dphase_err_rad"] = _masked_mean(dd, m_d * d.abs()).item()

    # Speckle contrast of intensity in tissue (1 for fully developed speckle; blur lowers it).
    for name, z in (("speckle_contrast", x_hat), ("speckle_contrast_gt", x)):
        i = z.abs() ** 2
        mu = _masked_mean(i, mask)
        var = _masked_mean((i - mu) ** 2, mask)
        out[name] = (var.sqrt() / (mu + _EPS)).item()
    return out
