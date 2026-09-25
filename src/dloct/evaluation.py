"""Shared evaluation loop: metrics for interpolation, model, and model + DC on full B-scans."""

from collections import defaultdict

import numpy as np
import torch

from .metrics import compute_metrics
from .physics import data_consistency, measure


def amp_dtype(precision: str = "auto"):
    """Autocast dtype for a precision setting; None means full fp32."""
    if precision == "fp32" or not torch.cuda.is_available():
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


@torch.no_grad()
def reconstruct(model, x, factor, offset=0, amp_dtype=None):
    """Returns {method: complex (B, Z, X)} for a clean complex batch ``x``."""
    x_meas = measure(x, factor, offset)
    with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None and x.is_cuda):
        x_hat = model(x_meas, factor, offset, apply_dc=False)
    out = {"interpolation": x_meas, "model": x_hat.to(torch.complex64)}
    if not getattr(model, "dc_builtin", False):
        out["model+dc"] = data_consistency(out["model"], x_meas, factor, offset)
    return out


def lateral_mps(z: torch.Tensor):
    """Mean lateral power spectrum of a complex (B, Z, X) batch, fftshifted, as numpy."""
    p = torch.fft.fftshift(torch.fft.fft(z, dim=-1, norm="ortho"), dim=-1).abs() ** 2
    return p.mean(dim=(0, 1)).cpu().numpy()


@torch.no_grad()
def evaluate(model, dataset, factor, device, snr_db=10.0, amp_dtype=None, keep=0, return_records=False):
    """
    Averages metrics over the B-scans of ``dataset`` (an ``EvalBScans``). Returns
    ``(summary, per_sample, examples, spectra)``: ``per_sample`` breaks the summary down by
    sample (A/B channels together), ``examples`` holds the first ``keep`` B-scans'
    reconstructions for figures and ``spectra`` the lateral MPS per method. With
    ``return_records`` it also returns the per-B-scan metrics
    ``[(sample, method, {metric: value}), ...]`` for confidence intervals.
    """
    model.eval()
    records = []
    mps = defaultdict(list)
    examples = []
    for i in range(len(dataset)):
        x, name, y = dataset[i]
        x = x.to(device)[None]
        recon = reconstruct(model, x, factor, amp_dtype=amp_dtype)
        sample = dataset.vols.volume(name)["group"].split("/", 1)[-1]
        for method, z in recon.items():
            records.append((sample, method, compute_metrics(z, x, snr_db, factor=factor)))
            mps[method].append(lateral_mps(z))
        mps["ground truth"].append(lateral_mps(x))
        if i < keep:
            examples.append((f"{name} y={y}", x[0].cpu(), {k: v[0].cpu() for k, v in recon.items()}))

    grouped = defaultdict(list)
    for sample, method, m in records:
        grouped[("all", method)].append(m)
        grouped[(sample, method)].append(m)
    table = defaultdict(dict)
    for (key, method), ms in grouped.items():
        table[key][method] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    width = min(len(v[0]) for v in mps.values())
    spectra = {}
    for method, lst in mps.items():
        same = [p for p in lst if len(p) == width]
        spectra[method] = (np.fft.fftshift(np.fft.fftfreq(width)), np.mean(same, axis=0))
    out = (dict(table["all"]), {k: dict(v) for k, v in table.items() if k != "all"}, examples, spectra)
    return out + (records,) if return_records else out


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap confidence interval of the mean over B-scans."""
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
