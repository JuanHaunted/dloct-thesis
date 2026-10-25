"""
Supporting thesis figures (same style as ``make_figures.py``), written to ``runs/figures/``:

  * ``data_coherence``: complex correlation between A-lines ``lag`` apart, raw vs bulk-phase
    corrected, per acquisition system. Source: the ``prepare_data`` log (``logs/slurm-dloct-prep-*.out``).
  * ``training_curves``: validation phase error and amplitude PSNR over training for unet_full and
    cascade_full, with the selected checkpoint marked. Source: ``runs/<run>/log.jsonl``.
  * ``k_comparison``: K=2 vs K=4 for interpolation, U-Net and cascade on three phase metrics.

    python scripts/make_extra_figures.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

OUT = Path("runs/figures")
OPHTHALMIC = ("Fovea", "OpticNerve", "S.Eye")   # in-vivo eye; everything else = benchtop system


def parse_prep_log(path: Path):
    rows, name = [], None
    for line in path.read_text().splitlines():
        m = re.match(r"(phase__\S+): \(Y,Z,X\)", line)
        if m:
            name = m.group(1).split("__", 1)[1]
            continue
        m = re.match(r"\s+(raw|bulk_corrected)\s+rho_lag1=([\d.]+) rho_lag2=([\d.]+) rho_lag4=([\d.]+)", line)
        if m and name:
            rows.append(dict(volume=name, kind=m.group(1), rho=[float(m.group(i)) for i in (2, 3, 4)]))
    return rows


def fig_data_coherence():
    logs = sorted(Path("logs").glob("slurm-dloct-prep-*.out"), key=lambda p: -len(p.read_text()))
    rows = parse_prep_log(logs[0])
    lags = np.array([1, 2, 4])
    groups = {"retina / optic nerve (ophthalmic SS-OCT, in vivo)": mf.VARIANT_COLOR["base"],
              "other tissues (benchtop system)": mf.VARIANT_COLOR["gan"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind, title in zip(axes, ("raw", "bulk_corrected"), ("Raw data", "After bulk-phase correction")):
        for (gname, color) in groups.items():
            sel = [r["rho"] for r in rows if r["kind"] == kind
                   and (r["volume"].startswith(OPHTHALMIC)) == gname.startswith("retina")]
            sel = np.array(sel)
            for v in sel:
                ax.plot(lags, v, color=color, linewidth=1, alpha=0.25)
            ax.plot(lags, np.median(sel, axis=0), color=color, linewidth=2.5, marker="o",
                    markeredgecolor=mf.SURFACE, markeredgewidth=2, label=f"{gname} (n={len(sel)}, median)")
        ax.set_xticks(lags, ["1", "2", "4"])
        ax.set_xlabel("distance between A-lines (lag)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("|complex correlation| (tissue pixels)")
    axes[1].annotate("K=2: missing lines are\n1 A-line from a measurement", (1, 0.62), xytext=(1.6, 0.86),
                     fontsize=8, color=mf.INK_2, arrowprops=dict(arrowstyle="-", color=mf.INK_2, linewidth=0.6))
    axes[1].annotate("K=4: up to 2 A-lines away", (2, 0.3), xytext=(2.6, 0.55), fontsize=8, color=mf.INK_2,
                     arrowprops=dict(arrowstyle="-", color=mf.INK_2, linewidth=0.6))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("Lateral coherence of the 29 volumes (thin lines: individual volumes)",
                 x=0.01, ha="left", fontsize=9, color=mf.INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    mf.save(fig, OUT, "data_coherence")


def fig_training_curves():
    runs = [("unet_full", "U-Net", mf.VARIANT_COLOR["base"], "-"),
            ("cascade_full", "Cascade", mf.VARIANT_COLOR["base"], "--")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, label, color, ls in runs:
        recs = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
        val = [r for r in recs if r["kind"] == "val"]
        steps = np.array([r["step"] for r in val]) / 1000
        key = "model+dc" if "model+dc" in val[0]["metrics"] else "model"
        phase = np.array([r["metrics"][key]["phase_err_w_rad"] for r in val])
        psnr = np.array([r["metrics"][key]["psnr_db"] for r in val])
        rho = np.array([r["metrics"][key]["rho_local"] for r in val])
        best = int(np.argmax(rho))
        for ax, y in zip(axes, (phase, psnr)):
            ax.plot(steps, y, color=color, linestyle=ls, linewidth=2, label=label)
            ax.scatter([steps[best]], [y[best]], s=50, color=color, edgecolors=mf.SURFACE, linewidths=2, zorder=3)
        interp = val[0]["metrics"]["interpolation"]
    axes[0].axhline(interp["phase_err_w_rad"], color=mf.BASELINE, linewidth=1)
    axes[0].annotate("interpolation", (steps[-1], interp["phase_err_w_rad"]), fontsize=8, color=mf.INK_2,
                     ha="right", va="bottom")
    axes[1].axhline(interp["psnr_db"], color=mf.BASELINE, linewidth=1)
    axes[1].annotate("interpolation", (steps[-1], interp["psnr_db"]), fontsize=8, color=mf.INK_2,
                     ha="right", va="bottom")
    axes[0].set_title("Validation phase error [rad] ↓", loc="left")
    axes[1].set_title("Validation PSNR, whole image [dB] ↑", loc="left")
    for ax in axes:
        ax.set_xlabel("training step [×1000]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation curves (Fovea4, OpticNerve3). Dots: checkpoint selected by local complex correlation. "
                 "Logged with the original −30 dB tissue mask.", x=0.01, ha="left", fontsize=8.5, color=mf.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    mf.save(fig, OUT, "training_curves")


def fig_k_comparison():
    sources = {2: ("unet_full", "cascade_full"), 4: ("unet_full_k4", "cascade_full_k4")}
    metrics = [("phase_err_w_rad", "phase error [rad] ↓"), ("dphase_err_rad", "Δφ error [rad] ↓"), ("ccc", "CCC ↑")]
    methods = [("Interpolation", "baseline"), ("U-Net", "base"), ("Cascade", "base")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    width = 0.24
    for ax, (key, title) in zip(axes, metrics):
        for j, (mname, variant) in enumerate(methods):
            vals = []
            for k, (unet, casc) in sources.items():
                run = casc if mname == "Cascade" else unet
                d = json.loads(Path(f"runs/{run}/eval_test_best_snr10/metrics.json").read_text())
                method = "interpolation" if mname == "Interpolation" else "model"
                vals.append(d["summary"][method][key])
            x = np.arange(2) + (j - 1) * width
            bars = ax.bar(x, vals, width * 0.9, color=mf.color_of(variant), alpha=0.85 if mname != "Cascade" else 0.5,
                          hatch="//" if mname == "Cascade" else None, edgecolor=mf.SURFACE, label=mname)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5, color=mf.INK_2)
        ax.set_xticks([0, 1], ["K = 2", "K = 4"])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Effect of the undersampling factor (test set, 5 volumes; tissue pixels ≥ 10 dB above noise)",
                 x=0.01, ha="left", fontsize=9, color=mf.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    mf.save(fig, OUT, "k_comparison")


if __name__ == "__main__":
    mf.style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_data_coherence()
    fig_training_curves()
    fig_k_comparison()
