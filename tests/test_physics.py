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


def test_bulk_phase_removal_is_independent_of_aline_jitter():
    # The correction removes any phase common to a whole A-line (jitter, bulk motion, and the
    # field's own depth-averaged step), so the result must not depend on the jitter.
    from dloct.prepare_data import remove_bulk_phase
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 64, 128)) + 1j * rng.standard_normal((2, 64, 128))).astype(np.complex64)
    jitter = np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 1, 128))).astype(np.complex64)
    clean, jittered = remove_bulk_phase(x.copy()), remove_bulk_phase(x * jitter)
    for y in range(2):   # equal up to one global phase per B-scan
        g = np.vdot(clean[y], jittered[y])
        assert np.allclose(jittered[y], clean[y] * g / abs(g), atol=1e-4)
    # Depth-dependent (local) phase structure survives: a phase ramp along z is untouched.
    ramp = np.exp(1j * 0.3 * np.arange(64))[None, :, None].astype(np.complex64)
    g = np.vdot(clean[0] * ramp[0], remove_bulk_phase(x * ramp)[0])
    assert np.allclose(remove_bulk_phase(x * ramp)[0], clean[0] * ramp[0] * g / abs(g), atol=1e-4)


@pytest.mark.parametrize("a,b", [
    ("OpticNerve3A", "OpticNerve3B"), ("OpticNerveANew", "OpticNerveBNew"),
    ("OpticNerveAOld", "OpticNerveBOld"), ("S.Eye2A", "S.Eye2B"), ("Fovea1A", "Fovea1B"),
    ("polInt1_polOut2_tomRaw", "polInt2_polOut1_tomRaw"),
])
def test_channels_of_one_sample_share_a_group(a, b):
    from dloct.prepare_data import group_of
    assert group_of("phase", a) == group_of("phase", b)


def test_distinct_samples_keep_distinct_groups():
    from dloct.prepare_data import group_of
    names = ["OpticNerve3A", "OpticNerve4A", "OpticNerveANew", "OpticNerveAOld", "unpairCadaverhearth", "ChickenBreastA"]
    assert len({group_of("phase", n) for n in names}) == len(names)


def test_phase_consistency_metrics_bounds_and_invariance():
    from dloct.metrics import phase_consistency_metrics
    x = rand_complex(1, 64, 64).to(torch.complex64)
    perfect = phase_consistency_metrics(x, x)
    assert all(abs(perfect[k] - 1) < 1e-4 for k in ("wpc", "ccc", "pg_ssim"))
    # CCC and PG-SSIM ignore a global phase offset; WPC does not (cos of the offset).
    rot = phase_consistency_metrics(x * torch.exp(torch.tensor(0.7j)).to(torch.complex64), x)
    assert abs(rot["ccc"] - 1) < 1e-4 and abs(rot["pg_ssim"] - 1) < 1e-4
    assert abs(rot["wpc"] - math.cos(0.7)) < 1e-4
    noise = phase_consistency_metrics(rand_complex(1, 64, 64, seed=5).to(torch.complex64), x)
    assert abs(noise["wpc"]) < 0.1 and noise["ccc"] < 0.1
