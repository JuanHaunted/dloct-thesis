"""
Reconstruction models: complex measurement (B, H, W) -> complex estimate (B, H, W).

* ``SingleShot``: one U-Net, residual on the sinc-interpolated measurement. Trained without
  data consistency (a final hard projection would zero the gradient on measured A-lines);
  DC is applied at inference only via ``apply_dc=True``.
* ``DCCascade``: unrolled physics-informed network (Schlemper et al. 2018). Alternates a
  small U-Net refinement with an exact data-consistency layer that re-inserts the acquired
  A-lines, and is trained end-to-end through the DC layers.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..physics import data_consistency, measured_mask, to_channels, to_complex
from .unet import UNet


def _net_input(x, factor, offset):
    b, h, w = x.shape
    mask = measured_mask(w, factor, offset, x.device).to(x.real.dtype)
    return torch.cat([to_channels(x), mask.expand(b, 1, h, w)], dim=1)


class SingleShot(nn.Module):
    def __init__(self, **unet_kwargs):
        super().__init__()
        self.net = UNet(in_channels=3, out_channels=2, **unet_kwargs)
        self.divisor = self.net.divisor

    def forward(self, x_meas, factor: int, offset: int = 0, apply_dc: bool = False):
        x = x_meas + to_complex(self.net(_net_input(x_meas, factor, offset)).float())
        return data_consistency(x, x_meas, factor, offset) if apply_dc else x


class DCCascade(nn.Module):
    dc_builtin = True

    def __init__(self, n_stages: int = 5, share_weights: bool = False,
                 grad_checkpoint: bool = False, **unet_kwargs):
        super().__init__()
        n_nets = 1 if share_weights else n_stages
        self.nets = nn.ModuleList(UNet(in_channels=3, out_channels=2, **unet_kwargs) for _ in range(n_nets))
        self.n_stages = n_stages
        self.grad_checkpoint = grad_checkpoint
        self.divisor = self.nets[0].divisor

    def forward(self, x_meas, factor: int, offset: int = 0, apply_dc: bool = True):
        # DC is part of every stage; ``apply_dc`` is accepted for a uniform interface.
        x = x_meas
        for i in range(self.n_stages):
            net = self.nets[i % len(self.nets)]
            inp = _net_input(x, factor, offset)
            if self.grad_checkpoint and self.training:
                out = checkpoint(net, inp, use_reentrant=False)
            else:
                out = net(inp)
            x = data_consistency(x + to_complex(out.float()), x_meas, factor, offset)
        return x


def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if "mults" in cfg:
        cfg["mults"] = tuple(cfg["mults"])
    if kind == "single_shot":
        return SingleShot(**cfg)
    if kind == "dc_cascade":
        return DCCascade(**cfg)
    raise ValueError(f"unknown model type {kind!r}")
