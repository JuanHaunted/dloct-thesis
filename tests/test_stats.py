import numpy as np

from dloct.stats import cluster_bootstrap_ci, decision_rule, holm, paired_comparison


def test_holm_matches_hand_computation():
    # p sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_cluster_bootstrap_is_wider_than_bscan_bootstrap_for_correlated_data():
    from dloct.stats import bootstrap_ci
    rng = np.random.default_rng(0)
    volume_effect = rng.normal(0, 1, 6)
    values = np.concatenate([v + 0.05 * rng.normal(size=50) for v in volume_effect])
    clusters = np.repeat(np.arange(6), 50)
    lo_b, hi_b = bootstrap_ci(values)
    lo_c, hi_c = cluster_bootstrap_ci(values, clusters)
    assert (hi_c - lo_c) > 3 * (hi_b - lo_b)


def _records(offsets, metric_shift):
    rng = np.random.default_rng(1)
    out = []
    for v in range(6):
        for y in range(20):
            base = 0.7 + 0.05 * rng.normal()
            out.append(dict(volume=f"v{v}", y=y, sample=f"s{v // 2}", method="m",
                            psnr_db_tissue=20 + offsets + rng.normal(0, 0.1),
                            wpc=base + metric_shift, ccc=base + metric_shift, pg_ssim=0.2,
                            nrmse=0.7 - metric_shift))
    return out


def test_paired_comparison_and_decision_rule():
    ref, better = _records(0.0, 0.0), _records(0.5, 0.05)
    rows = paired_comparison(ref, better, ["psnr_db_tissue", "wpc", "ccc", "pg_ssim", "nrmse"])
    by = {r["metric"]: r for r in rows}
    assert by["wpc"]["win_rate"] == 1.0 and by["nrmse"]["win_rate"] == 1.0
    assert by["wpc"]["p_bscan_holm"] < 1e-6
    assert decision_rule(rows, "psnr_db_tissue")["passes"]
    worse_phase = _records(0.5, -0.1)
    verdict = decision_rule(paired_comparison(ref, worse_phase, ["psnr_db_tissue", "wpc", "ccc", "pg_ssim"]),
                            "psnr_db_tissue")
    assert not verdict["passes"] and "wpc" in verdict["phase_degraded"]
