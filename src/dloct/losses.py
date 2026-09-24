"""
Phase-aware losses for complex OCT reconstruction (docs/literature_review.md §6).

All inputs are complex tensors (B, H, W), lateral axis last. Every term is invariant to a
global phase rotation of prediction and target together, so no absolute phase is preferred.

Why the phase terms exist: |x̂ − x|² = (|x̂| − |x|)² + 2|x̂||x|(1 − cos Δφ). A plain complex loss
already supervises phase, but weighted by amplitude², so dim tissue gets almost none.
"""

import torch
import torch.nn.functional as F

_DELTA = 1e-6


def complex_charbonnier(x_hat, x, eps: float = 1e-3):
    """Rotation-invariant complex L1: mean sqrt(|x̂ − x|² + ε²)."""
    d = x_hat - x
    return torch.sqrt(d.real ** 2 + d.imag ** 2 + eps ** 2).mean()


def magnitude_l1(x_hat, x, eps: float = 1e-8):
    """Amplitude-only L1 (ablation: the loss most prior OCT/MRI work uses)."""
    amp_hat = torch.sqrt(x_hat.real ** 2 + x_hat.imag ** 2 + eps)
    return (amp_hat - x.abs()).abs().mean()


def log_magnitude(x_hat, x, floor: float = 1e-2):
    """
    L1 between log amplitudes, |log((|x̂| + c) / (|x| + c))|, c ≈ −40 dB of the P99.9 amplitude.

    Complex losses reward the conditional mean, whose magnitude shrinks wherever phase is
    uncertain, so unmeasured A-lines come out dark. This term penalizes that shrinkage in
    the dB domain the amplitude is judged in.
    """
    amp_hat = torch.sqrt(x_hat.real ** 2 + x_hat.imag ** 2 + 1e-12)
    return (torch.log(amp_hat + floor) - torch.log(x.abs() + floor)).abs().mean()


def _cos_dphi(a, b):
    """cos of the phase difference between a and b, computed without atan2."""
    return (a * b.conj()).real / (a.abs() * b.abs() + _DELTA)


def weighted_phase(x_hat, x):
    """Σ w (1 − cos Δφ) / Σ w, with w = |x| (linear, not quadratic, amplitude weight)."""
    w = x.abs()
    return (w * (1 - _cos_dphi(x_hat, x))).sum() / (w.sum() + _DELTA)


def _box3(z):
    """3x3 box sum of a complex (B, H, W) field."""
    k = torch.ones(1, 1, 3, 3, device=z.device, dtype=z.real.dtype)
    re = F.conv2d(z.real.unsqueeze(1), k, padding=1).squeeze(1)
    im = F.conv2d(z.imag.unsqueeze(1), k, padding=1).squeeze(1)
    return torch.complex(re, im)


def lateral_phasor(u, smooth: bool = True):
    """Inter-A-line phasor u[x+1]·conj(u[x]) (Kasai / phase-resolved Doppler)."""
    d = u[..., 1:] * u[..., :-1].conj()
    return _box3(d) if smooth else d


def phase_difference(x_hat, x, smooth: bool = True):
    """Amplitude-weighted wrapped error of the inter-A-line phase difference."""
    d_hat, d = lateral_phasor(x_hat, smooth), lateral_phasor(x, smooth)
    w = d.abs()
    return (w * (1 - _cos_dphi(d_hat, d))).sum() / (w.sum() + _DELTA)


def lateral_spectrum_l1(x_hat, x):
    """L1 on the complex difference of unitary lateral spectra (targets the folded band)."""
    d = torch.fft.fft(x_hat - x, dim=-1, norm="ortho")
    return d.abs().mean()


class ReconLoss(torch.nn.Module):
    """
    L = w_c·L_c + w_mag·L_mag + w_logmag·L_logmag + w_phase·L_φ + w_dphase·L_Δφ + w_fft·L_F.

    ``warmup_steps`` linearly ramps the two phase terms from 0 so they do not dominate
    before the amplitude is roughly right.
    """

    def __init__(self, charbonnier=1.0, magnitude=0.0, log_magnitude=0.0, phase=0.1, dphase=0.1,
                 fft=0.01, eps=1e-3, warmup_steps=0):
        super().__init__()
        self.w = dict(charbonnier=charbonnier, magnitude=magnitude, log_magnitude=log_magnitude,
                      phase=phase, dphase=dphase, fft=fft)
        self.eps = eps
        self.warmup_steps = warmup_steps

    def forward(self, x_hat, x, step: int = 0):
        ramp = min(1.0, step / self.warmup_steps) if self.warmup_steps else 1.0
        terms = {}
        if self.w["charbonnier"]:
            terms["charbonnier"] = complex_charbonnier(x_hat, x, self.eps)
        if self.w["magnitude"]:
            terms["magnitude"] = magnitude_l1(x_hat, x)
        if self.w["log_magnitude"]:
            terms["log_magnitude"] = log_magnitude(x_hat, x)
        if self.w["phase"]:
            terms["phase"] = weighted_phase(x_hat, x)
        if self.w["dphase"]:
            terms["dphase"] = phase_difference(x_hat, x)
        if self.w["fft"]:
            terms["fft"] = lateral_spectrum_l1(x_hat, x)
        scale = {"phase": ramp, "dphase": ramp}
        total = sum(self.w[k] * scale.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: v.detach() for k, v in terms.items()}
