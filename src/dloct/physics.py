"""
Lateral sub-Nyquist sampling operator for complex OCT B-scans (README operator C).

All functions act on complex tensors whose **last axis is the lateral (fast) axis** X.
A measurement keeps every K-th A-line starting at ``offset`` (no anti-alias filter, exactly
what a scanner that acquires fewer A-lines does) and sinc-interpolates it back onto the
fine grid, so the network sees the aliased field at full size. The measured A-lines are
preserved exactly by the interpolation, which makes data consistency a plain overwrite.
"""

import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) real Re/Im channels -> (B, H, W) complex."""
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def to_channels(z: torch.Tensor) -> torch.Tensor:
    """(B, H, W) complex -> (B, 2, H, W) real Re/Im channels."""
    return torch.stack([z.real, z.imag], dim=1)


def sinc_upsample(y: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Band-limited (sinc) interpolation along the last axis by an integer factor, via
    zero-padding the centered spectrum. Input samples land on output indices ``::factor``
    unchanged. For even input length the unpaired Nyquist bin is split across both band
    edges so the interpolant is symmetric.
    """
    if factor == 1:
        return y.clone()
    m = y.shape[-1]
    n = m * factor
    spec = torch.fft.fftshift(torch.fft.fft(y, dim=-1), dim=-1)
    lo = n // 2 - m // 2
    padded = torch.zeros(*y.shape[:-1], n, dtype=spec.dtype, device=spec.device)
    padded[..., lo:lo + m] = spec
    if m % 2 == 0:
        half = padded[..., lo] * 0.5
        padded[..., lo] = half
        padded[..., lo + m] = half
    return torch.fft.ifft(torch.fft.ifftshift(padded, dim=-1), dim=-1) * factor


def measured_mask(width: int, factor: int, offset: int, device=None) -> torch.Tensor:
    """Boolean (W,) mask of acquired A-line positions."""
    mask = torch.zeros(width, dtype=torch.bool, device=device)
    mask[offset::factor] = True
    return mask


def measure(x: torch.Tensor, factor: int, offset: int = 0) -> torch.Tensor:
    """
    Simulate lateral decimation by ``factor`` and sinc-interpolate back to the fine grid.

    ``x`` is complex (..., W) with W divisible by ``factor``. Returns complex (..., W) whose
    samples at ``offset::factor`` equal ``x`` there exactly.
    """
    w = x.shape[-1]
    if w % factor:
        raise ValueError(f"lateral width {w} not divisible by factor {factor}")
    if factor == 1:
        return x.clone()
    shifted = torch.roll(x, -offset, dims=-1)
    y = sinc_upsample(shifted[..., ::factor], factor)
    return torch.roll(y, offset, dims=-1)


def data_consistency(x_hat: torch.Tensor, x_meas: torch.Tensor, factor: int,
                     offset: int = 0) -> torch.Tensor:
    """
    Exact projection onto the measurement-consistent set: overwrite the acquired A-lines
    of the estimate with the measured ones. Works on complex (..., W) or on real
    channel tensors (..., W) alike.
    """
    out = x_hat.clone()
    out[..., offset::factor] = x_meas[..., offset::factor]
    return out
