"""
Amplitude and phase fidelity metrics (docs/literature_review.md §5).

Inputs are complex tensors (B, H, W), lateral axis last, in normalized units (the per-volume
99.9th-percentile amplitude is 1). Phase metrics are restricted to a signal mask: pixels at
least ``snr_db`` above the B-scan's noise floor. Phase in noise is random and is not scored.
"""

import math

import torch
import torch.nn.functional as F

from .losses import lateral_phasor

_EPS = 1e-12


def noise_floor(x):
    """
    Per-B-scan noise amplitude of a complex (B, Z, X) batch: the median amplitude of the 10 %
    of depth rows with the lowest median amplitude (rows above the tissue or deep below it).
    """
    row = x.abs().median(dim=-1).values                  # (B, Z)
    k = max(1, row.shape[-1] // 10)
    return row.sort(dim=-1).values[:, :k].median(dim=-1).values   # (B,)


def tissue_mask(x, snr_db: float = 10.0):
    """Pixels at least ``snr_db`` above each B-scan's noise floor."""
    return x.abs() > noise_floor(x)[:, None, None] * 10 ** (snr_db / 20)


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


def psnr(a, b, mask=None):
    """PSNR of [0, 1] images (B, H, W), optionally over a boolean mask only."""
    if mask is None:
        mse = ((a - b) ** 2).mean(dim=(-2, -1))
    else:
        m = mask.float()
        mse = (((a - b) ** 2) * m).sum(dim=(-2, -1)) / (m.sum(dim=(-2, -1)) + _EPS)
    return (10 * torch.log10(1.0 / (mse + _EPS))).mean()


def ssim(a, b, window: int = 11, sigma: float = 1.5, data_range: float = 1.0, mask=None):
    """
    Gaussian-window SSIM of images (B, H, W) with the given dynamic range. Without ``mask`` it
    is the mean over valid windows; with a mask (B, H, W) it is the mean of the same-size SSIM
    map over the masked pixels.
    """
    g = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    pad = 0 if mask is None else window // 2
    a, b = a.unsqueeze(1), b.unsqueeze(1)
    conv = lambda v: F.conv2d(v, k, padding=pad)
    mu_a, mu_b = conv(a), conv(b)
    var_a = conv(a * a) - mu_a ** 2
    var_b = conv(b * b) - mu_b ** 2
    cov = conv(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    if mask is None:
        return s.mean()
    return _masked_mean(s.squeeze(1), mask.float())


def hist_similarity(a, b, bins: int = 256):
    """Cosine similarity of the 256-bin histograms of two [0, 1] images (1 = same distribution)."""
    ha = torch.histc(a.float(), bins=bins, min=0, max=1)
    hb = torch.histc(b.float(), bins=bins, min=0, max=1)
    return (ha @ hb / (ha.norm() * hb.norm() + _EPS)).item()


def coherence_by_decile(x_hat, x, n_bins: int = 10):
    """
    Mean cos(φ̂ − φ) within each decile of ground-truth amplitude (d1 = weakest, noise;
    d10 = strongest signal). Unweighted, so it shows where along the signal range phase is kept.
    """
    amp = x.abs().flatten()
    cosd = ((x_hat * x.conj()).real / (x_hat.abs() * x.abs() + _EPS)).flatten()
    edges = torch.quantile(amp.float(), torch.linspace(0, 1, n_bins + 1, device=amp.device))
    out = {}
    for i in range(n_bins):
        hi_ok = amp <= edges[i + 1] if i == n_bins - 1 else amp < edges[i + 1]
        m = (amp >= edges[i]) & hi_ok
        out[f"coh_d{i + 1}"] = cosd[m].mean().item() if m.any() else float("nan")
    return out


def _phase_gradients(z):
    """Wrapped axial and lateral phase differences of a complex (B, Z, X) field."""
    return torch.angle(z[:, 1:] * z[:, :-1].conj()), torch.angle(z[..., 1:] * z[..., :-1].conj())


def phase_consistency_metrics(x_hat, x):
    """
    Complex-field phase metrics that are insensitive to the absolute phase:

    * ``wpc``: amplitude-weighted phase coherence Σ|x̂||x|cos Δφ / Σ|x̂||x| over the B-scan,
      in [-1, 1] (1 = aligned, 0 = decorrelated).
    * ``ccc``: complex coherence |Σ x̂ x̄| / sqrt(Σ|x̂|² Σ|x|²) over pixels brighter than the
      B-scan's median ground-truth amplitude (invariant to a global phase offset).
    * ``pg_ssim``: mean SSIM (range 2π) of the wrapped axial and lateral phase-gradient maps,
      averaged over the same bright-pixel mask.
    """
    cross = x_hat * x.conj()
    wpc = cross.real.sum(dim=(-2, -1)) / ((x_hat.abs() * x.abs()).sum(dim=(-2, -1)) + _EPS)
    amp = x.abs()
    bright = amp > amp.flatten(1).median(dim=1).values[:, None, None]
    m = bright.float()
    ccc = (cross * m).sum(dim=(-2, -1)).abs() / torch.sqrt(
        (x_hat.abs() ** 2 * m).sum(dim=(-2, -1)) * (amp ** 2 * m).sum(dim=(-2, -1)) + _EPS)
    (gz_hat, gx_hat), (gz, gx) = _phase_gradients(x_hat), _phase_gradients(x)
    pg = 0.5 * (ssim(gz_hat, gz, data_range=2 * math.pi, mask=bright[:, 1:] & bright[:, :-1])
                + ssim(gx_hat, gx, data_range=2 * math.pi, mask=bright[..., 1:] & bright[..., :-1]))
    return {"wpc": wpc.mean().item(), "ccc": ccc.mean().item(), "pg_ssim": pg.item()}


@torch.no_grad()
def compute_metrics(x_hat, x, snr_db: float = 10.0, local_window: int = 5, factor: int | None = None,
                    offset: int = 0):
    """
    Returns a dict of floats. ``x_hat`` and ``x`` are complex (B, H, W). With ``factor`` (and
    ``offset``) it also reports amplitude fidelity on the unmeasured A-lines in tissue.
    """
    x_hat, x = x_hat.to(torch.complex64), x.to(torch.complex64)
    mask = tissue_mask(x, snr_db).float()
    out_mask = {"mask_fraction": mask.mean().item()}
    out = {}

    # Amplitude, dB domain.
    a_hat, a = to_db_image(x_hat), to_db_image(x)
    out["psnr_db"] = psnr(a_hat, a).item()
    out["ssim_db"] = ssim(a_hat, a).item()
    # Whole-image dB metrics are dominated by background noise (most pixels): report tissue too.
    out["hist_sim"] = hist_similarity(a_hat, a)
    out["psnr_db_tissue"] = psnr(a_hat, a, mask.bool()).item()
    out["ssim_db_tissue"] = ssim(a_hat, a, mask=mask.bool()).item()
    if factor and factor > 1:
        # Amplitude on the unmeasured A-lines in tissue: a mean dB bias below 0 and a power
        # ratio (unmeasured / measured) below the ground truth's mean the fill-in is too dark,
        # which shows as vertical striping.
        cols = torch.arange(x.shape[-1], device=x.device)
        missing = ((cols - offset) % factor != 0)[None, None, :] & mask.bool()
        measured = ((cols - offset) % factor == 0)[None, None, :] & mask.bool()
        db = lambda z: 20 * torch.log10(z.abs() + _EPS)
        out["unmeasured_db_bias"] = ((db(x_hat) - db(x))[missing]).mean().item()
        power = lambda z, m: (z.abs() ** 2)[m].mean()
        out["unmeasured_power_ratio"] = (power(x_hat, missing) / (power(x_hat, measured) + _EPS)).item()
        out["unmeasured_power_ratio_gt"] = (power(x, missing) / (power(x, measured) + _EPS)).item()

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
    out.update(phase_consistency_metrics(x_hat, x))
    out.update(coherence_by_decile(x_hat, x))
    out.update(out_mask)
    return out
