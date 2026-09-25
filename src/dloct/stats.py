"""
Statistics for comparing reconstructions over paired test B-scans.

B-scans from one volume are strongly correlated, so two kinds of uncertainty are reported:

* ``bootstrap_ci``: percentile bootstrap of the mean over B-scans (optimistic, treats B-scans
  as independent);
* ``cluster_bootstrap_ci``: resamples whole volumes and averages their per-volume means
  (honest about the correlation, but wide when there are few volumes).

Paired comparisons use the Wilcoxon signed-rank test (non-parametric, no normality assumption),
both over B-scans and over per-volume means, with Holm–Bonferroni correction across the
family of tests.
"""

from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean, treating values as independent."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def cluster_means(values, clusters):
    """Per-cluster means, in a fixed (sorted) cluster order."""
    groups = defaultdict(list)
    for v, c in zip(values, clusters):
        if np.isfinite(v):
            groups[c].append(v)
    keys = sorted(groups)
    return keys, np.array([np.mean(groups[k]) for k in keys])


def cluster_bootstrap_ci(values, clusters, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Bootstrap CI of the mean of per-cluster (per-volume) means, resampling clusters."""
    _, means = cluster_means(values, clusters)
    return bootstrap_ci(means, n_boot, level, seed)


def holm(pvalues):
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adjusted[i] = running
    return adjusted


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank p-value for paired samples; NaN if undefined."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.all(a == b):
        return float("nan")
    return float(wilcoxon(a, b).pvalue)


# Direction of improvement per metric: +1 higher is better, -1 lower is better, 0 closer to
# zero is better (compared on the absolute value).
DIRECTION = {
    "psnr_db": 1, "ssim_db": 1, "psnr_db_tissue": 1, "ssim_db_tissue": 1, "hist_sim": 1,
    "nrmse": -1, "rho_global": 1, "rho_local": 1,
    "wpc": 1, "ccc": 1, "pg_ssim": 1,
    "phase_err_rad": -1, "phase_err_w_rad": -1, "dphase_err_rad": -1,
    "unmeasured_db_bias": 0,
}


def _oriented(values, metric):
    v = np.asarray(values, dtype=np.float64)
    return np.abs(v) if DIRECTION.get(metric, 1) == 0 else v


def paired_comparison(ref_records, cand_records, metrics, n_boot: int = 2000, seed: int = 0):
    """
    Compare a candidate against a reference on the B-scans they share, paired by (volume, y).

    For each metric returns the mean of both, the mean paired difference (candidate − reference)
    with a volume-level bootstrap CI, the fraction of B-scans where the candidate is better, and
    two Wilcoxon p-values: over B-scan pairs and over per-volume means (the conservative one; it
    needs >= 6 volumes to reach p < 0.05). Holm-adjusted p-values are added across ``metrics``.
    """
    ref = {(r["volume"], r["y"]): r for r in ref_records}
    pairs = [(ref[(c["volume"], c["y"])], c) for c in cand_records if (c["volume"], c["y"]) in ref]
    if not pairs:
        raise ValueError("no shared B-scans between reference and candidate")
    volumes = [c["volume"] for _, c in pairs]
    rows = []
    for m in metrics:
        a = _oriented([r[m] for r, _ in pairs], m)
        b = _oriented([c[m] for _, c in pairs], m)
        d = b - a
        sign = -1 if DIRECTION.get(m, 1) in (-1, 0) else 1
        _, va = cluster_means(a, volumes)
        _, vb = cluster_means(b, volumes)
        rows.append(dict(
            metric=m, n_bscans=len(pairs), n_volumes=len(va),
            ref=float(np.nanmean(a)), cand=float(np.nanmean(b)), delta=float(np.nanmean(d)),
            delta_ci_volume=cluster_bootstrap_ci(d, volumes, n_boot, seed=seed),
            win_rate=float(np.nanmean(sign * d > 0)),
            p_bscan=paired_wilcoxon(a, b), p_volume=paired_wilcoxon(va, vb),
        ))
    for key in ("p_bscan", "p_volume"):
        for row, adj in zip(rows, holm([r[key] for r in rows])):
            row[key + "_holm"] = float(adj)
    return rows


def decision_rule(rows, main_metric: str, phase_metrics=("wpc", "ccc", "pg_ssim"), tol: float = 0.05,
                  p_key: str = "p_bscan_holm", alpha: float = 0.05):
    """
    Pre-registered acceptance rule for "the candidate is better than the reference":
    (1) it improves ``main_metric``; (2) no phase metric degrades by more than ``tol`` (relative);
    (3) WPC or CCC improves with Holm-adjusted p < ``alpha``.
    """
    by = {r["metric"]: r for r in rows}

    def improves(r):
        return (r["cand"] - r["ref"]) * (-1 if DIRECTION.get(r["metric"], 1) in (-1, 0) else 1) > 0

    main_ok = improves(by[main_metric])
    degraded = [m for m in phase_metrics if m in by and by[m]["ref"] != 0
                and (by[m]["cand"] - by[m]["ref"]) / abs(by[m]["ref"]) < -tol]
    significant = [m for m in ("wpc", "ccc") if m in by and improves(by[m]) and by[m][p_key] < alpha]
    return dict(passes=bool(main_ok and not degraded and significant), main_improves=bool(main_ok),
                phase_degraded=degraded, significant_phase_gain=significant)


def comparison_table(rows) -> str:
    """Markdown table of ``paired_comparison`` rows."""
    head = ("| metric | reference | candidate | Δ [95% CI, by volume] | better on | "
            "p (B-scans, Holm) | p (volumes, Holm) |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lo, hi = r["delta_ci_volume"]
        name = f"\\|{r['metric']}\\|" if DIRECTION.get(r["metric"], 1) == 0 else r["metric"]
        lines.append(f"| {name} | {r['ref']:.4f} | {r['cand']:.4f} | {r['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{r['win_rate']:.0%} | {r['p_bscan_holm']:.2g} | {r['p_volume_holm']:.2g} |")
    return "\n".join(lines)
