"""Shared evaluation loop: metrics for interpolation, model, and model + DC on full B-scans."""

from collections import defaultdict

import numpy as np
import torch

from .metrics import compute_metrics, ssim, to_db_image
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
    ``return_records`` it also returns the per-B-scan metrics as dicts with keys ``sample``,
    ``volume``, ``y``, ``method`` and the metric values, for statistics.

    ``identity_ssim`` is the SSIM (dB image) between a method's output and the interpolated
    measurement it started from: close to 1 means the method barely changed its input.
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
        meas_db = to_db_image(recon["interpolation"])
        for method, z in recon.items():
            m = compute_metrics(z, x, snr_db, factor=factor)
            m["identity_ssim"] = ssim(to_db_image(z), meas_db).item()
            records.append(dict(sample=sample, volume=name, y=int(y), method=method, **m))
            mps[method].append(lateral_mps(z))
        mps["ground truth"].append(lateral_mps(x))
        if i < keep:
            examples.append((f"{name} y={y}", x[0].cpu(), {k: v[0].cpu() for k, v in recon.items()}))

    grouped = defaultdict(list)
    for r in records:
        grouped[("all", r["method"])].append(r)
        grouped[(r["sample"], r["method"])].append(r)
    table = defaultdict(dict)
    for (key, method), rs in grouped.items():
        table[key][method] = {k: float(np.nanmean([r[k] for r in rs])) for k in metric_keys(rs[0])}
    width = min(len(v[0]) for v in mps.values())
    spectra = {}
    for method, lst in mps.items():
        same = [p for p in lst if len(p) == width]
        spectra[method] = (np.fft.fftshift(np.fft.fftfreq(width)), np.mean(same, axis=0))
    out = (dict(table["all"]), {k: dict(v) for k, v in table.items() if k != "all"}, examples, spectra)
    return out + (records,) if return_records else out


RECORD_KEYS = ("sample", "volume", "y", "method")


def metric_keys(record: dict):
    """The metric names of a per-B-scan record (everything except its identifiers)."""
    return [k for k in record if k not in RECORD_KEYS]
