"""
Compare two evaluated runs (or two methods of one run) on the same test B-scans.

    python -m dloct.compare --ref runs/unet_full/eval_test_best_snr10 \\
                            --cand runs/unet_gan/eval_test_best_snr10 [--main psnr_db_tissue]

Pairs B-scans by (volume, y), runs paired Wilcoxon tests (Holm-corrected) and applies the
pre-registered decision rule in ``dloct.stats.decision_rule``. Writes ``compare_<ref>_vs_<cand>.md``
next to the candidate's evaluation.
"""

import argparse
import json
from pathlib import Path

from .stats import comparison_table, decision_rule, paired_comparison

METRICS = ["psnr_db_tissue", "ssim_db_tissue", "hist_sim", "unmeasured_db_bias", "nrmse", "rho_local",
           "wpc", "ccc", "pg_ssim", "phase_err_w_rad", "dphase_err_rad"]


def load(path: Path, method: str):
    data = json.loads((path / "metrics.json").read_text())
    records = [r for r in data["per_bscan"] if r["method"] == method]
    if not records:
        raise SystemExit(f"{path}: no records for method {method!r}")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="evaluation directory of the reference")
    p.add_argument("--cand", required=True, help="evaluation directory of the candidate")
    p.add_argument("--ref-method", default="model")
    p.add_argument("--cand-method", default="model")
    p.add_argument("--main", default="psnr_db_tissue", help="main metric of the decision rule")
    args = p.parse_args()

    ref, cand = Path(args.ref), Path(args.cand)
    rows = paired_comparison(load(ref, args.ref_method), load(cand, args.cand_method), METRICS)
    verdict = decision_rule(rows, args.main)
    title = f"{ref.parent.name}/{args.ref_method} (reference) vs {cand.parent.name}/{args.cand_method} (candidate)"
    md = [f"# {title}", "", comparison_table(rows), "",
          "Decision rule (pre-registered): the candidate improves the main metric "
          f"(`{args.main}`), no phase metric (WPC, CCC, PG-SSIM) degrades by more than 5%, and WPC or "
          "CCC improves with Holm-adjusted p < 0.05 over B-scans.", "",
          f"**Passes: {verdict['passes']}**. Main metric improves: {verdict['main_improves']}; "
          f"degraded phase metrics: {verdict['phase_degraded'] or 'none'}; "
          f"significant phase gains: {verdict['significant_phase_gain'] or 'none'}."]
    out = cand / f"compare_{ref.parent.name}_{args.ref_method}_vs_{cand.parent.name}_{args.cand_method}.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
