import math

import numpy as np
import pytest
import torch

from dloct.losses import ReconLoss
from dloct.models.reconstructors import DCCascade, SingleShot
from dloct.physics import data_consistency, measure, sinc_upsample
from dloct.sampling_analysis import subsample_lateral


def rand_complex(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                         torch.randn(*shape, generator=g, dtype=torch.float64))


@pytest.mark.parametrize("factor", [2, 4, 8])
@pytest.mark.parametrize("offset", [0, 1])
def test_measure_keeps_acquired_alines(factor, offset):
    x = rand_complex(3, 16, 64)
    y = measure(x, factor, offset)
    assert torch.allclose(y[..., offset::factor], x[..., offset::factor], atol=1e-12)


def test_measure_matches_numpy_reference():
    x = rand_complex(8, 64)
    ref = subsample_lateral(x.numpy(), 4, axis=1, interpolate=True)
    assert np.allclose(measure(x, 4).numpy(), ref, atol=1e-12)


def test_bandlimited_signal_is_recovered_exactly():
    n = torch.arange(64, dtype=torch.float64)
    x = torch.exp(2j * math.pi * 5 * n / 64)[None]   # inside the K=4 band (|f| < 8/64)
    assert torch.allclose(measure(x, 4), x, atol=1e-12)


def test_aliasing_happens_out_of_band():
    n = torch.arange(64, dtype=torch.float64)
    x = torch.exp(2j * math.pi * 20 * n / 64)[None]  # outside the K=4 band -> folds
    assert (measure(x, 4) - x).abs().max() > 0.5


def test_sinc_upsample_identity_factor_one():
    x = rand_complex(2, 32)
    assert torch.allclose(sinc_upsample(x, 1), x)


def test_data_consistency_is_a_projection():
    x, x_hat = rand_complex(2, 8, 32, seed=1), rand_complex(2, 8, 32, seed=2)
    y = measure(x, 4, 1)
    once = data_consistency(x_hat, y, 4, 1)
    assert torch.allclose(once[..., 1::4], x[..., 1::4])
    assert torch.allclose(data_consistency(once, y, 4, 1), once)


def test_losses_zero_at_target_and_phase_invariant():
    x = rand_complex(2, 32, 32).to(torch.complex64)
    x_hat = x + 0.1 * rand_complex(2, 32, 32, seed=3).to(torch.complex64)
    loss = ReconLoss(eps=1e-8)
    assert loss(x, x)[0].item() < 1e-5
    rot = torch.exp(torch.tensor(1.234j)).to(torch.complex64)
    a, b = loss(x_hat, x)[0], loss(x_hat * rot, x * rot)[0]
    assert torch.allclose(a, b, rtol=1e-4)


@pytest.mark.parametrize("cls,kw", [(SingleShot, {}), (DCCascade, dict(n_stages=2, base=16))])
def test_models_start_at_measurement_and_shapes(cls, kw):
    model = cls(mults=(1, 2, 4), **kw)
    x = rand_complex(2, 32, 64).to(torch.complex64)
    y = measure(x, 4, 2)
    out = model(y, 4, 2)
    assert out.shape == y.shape and out.dtype == torch.complex64
    assert torch.allclose(out, y, atol=1e-6)  # zero-initialized head => identity at init
    model(y, 4, 2).abs().mean().backward()
