"""
Conditional PatchGAN discriminator for speckle realism.

It judges overlapping patches (receptive field ~70 px for 3 layers) of a candidate B-scan given
the measurement, so it can learn what fully developed speckle looks like, including on the
unmeasured A-lines (the mask channel tells it which lines were acquired). Spectral normalization
on every conv keeps adversarial training stable without batch norm. Trained with the hinge loss.

Input modes:
* ``amplitude`` (default): dB amplitude of candidate and measurement, plus the mask (3 channels).
  Realism pressure acts on amplitude only; phase stays governed by the physics-aware losses.
* ``logphasor``: dB amplitude times e^{iφ} as two channels, for candidate and measurement, plus the
  mask (5 channels). The discriminator also sees phase texture.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from ..metrics import to_db_image
from ..physics import measured_mask

N_CHANNELS = {"amplitude": 3, "logphasor": 5}


def _features(z, mode):
    db = to_db_image(z)
    if mode == "amplitude":
        return db[:, None]
    phasor = z / (z.abs() + 1e-12)
    return torch.stack([db * phasor.real, db * phasor.imag], dim=1)


def disc_input(candidate, x_meas, factor: int, offset: int, mode: str = "amplitude"):
    """Discriminator input for a complex candidate (B, Z, X) conditioned on the measurement."""
    b, h, w = candidate.shape
    mask = measured_mask(w, factor, offset, candidate.device).to(candidate.real.dtype)
    return torch.cat([_features(candidate, mode), _features(x_meas, mode), mask.expand(b, 1, h, w)], dim=1)


class PatchDiscriminator(nn.Module):
    def __init__(self, mode: str = "amplitude", base: int = 64, n_layers: int = 3):
        super().__init__()
        self.mode = mode
        sn = spectral_norm
        layers = [sn(nn.Conv2d(N_CHANNELS[mode], base, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(1, n_layers):
            nxt = min(base * 2 ** i, base * 8)
            layers += [sn(nn.Conv2d(ch, nxt, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True)]
            ch = nxt
        nxt = min(ch * 2, base * 8)
        layers += [sn(nn.Conv2d(ch, nxt, 4, stride=1, padding=1)), nn.LeakyReLU(0.2, True),
                   sn(nn.Conv2d(nxt, 1, 4, stride=1, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, candidate, x_meas, factor: int, offset: int):
        return self.net(disc_input(candidate, x_meas, factor, offset, self.mode))


def d_hinge_loss(real_logits, fake_logits):
    return torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()
