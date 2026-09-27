"""
Thesis figures comparing all evaluated models (static PNG + PDF, light background).

Quantitative figures read only ``runs/<run>/eval_test_<ckpt>_snr10/metrics.json``:
  * ``tradeoff``: amplitude vs phase fidelity per model (tissue PSNR vs φ error; HistSim vs Δφ
    error), means with volume-level 95% CIs;
  * ``boxplots``: per-B-scan distributions of the key metrics, one panel per metric;
  * ``deciles``: phase coherence by ground-truth amplitude decile.

Qualitative figures read ``runs/figures/recon_<sample>.npz`` written on the cluster by
``scripts/dump_reconstructions.py``:
  * ``qual_amplitude_<sample>``: B-scan amplitude per model with two zoomed regions;
  * ``qual_phase_<sample>``: axial and lateral phase-difference maps per model.

    python scripts/make_figures.py [--out runs/figures] [--only tradeoff boxplots deciles qual]

Encoding, fixed across figures: colour = training variant, marker = architecture, grey =
interpolation. Identity never relies on colour alone (every mark is labelled).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dloct.stats import cluster_bootstrap_ci  # noqa: E402

# Validated categorical order (light surface), slots 1..6; grey for the baseline.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BASELINE = "#8a8983"
VARIANT_COLOR = {
    "base": "#2a78d6",           # blue
    "gan": "#eb6834",            # orange
    "power": "#1baf7a",          # aqua
    "magnitude": "#eda100",      # yellow
    "no_phase": "#e87ba4",       # magenta
    "gan_power": "#008300",      # green
}
ARCH_MARKER = {"unet": "o", "cascade": "s", "none": "D"}

# (label, run, ckpt, method, variant, arch). Missing evaluations are skipped.
MODELS = [
    ("Interpolation", "unet_full", "best", "interpolation", "baseline", "none"),
    ("U-Net", "unet_full", "best", "model", "base", "unet"),
    ("Cascade", "cascade_full", "best", "model", "base", "cascade"),
    ("U-Net + D", "unet_gan", "latest", "model", "gan", "unet"),
    ("Cascade + D", "cascade_gan", "latest", "model", "gan", "cascade"),
    ("U-Net + power", "unet_power", "latest", "model", "power", "unet"),
    ("U-Net + D + power", "unet_gan_power", "latest", "model", "gan_power", "unet"),
    ("U-Net amplitude-only", "unet_magnitude", "best", "model", "magnitude", "unet"),
    ("U-Net no phase terms", "unet_complex", "best", "model", "no_phase", "unet"),
]
EXCLUDED_VOLUMES = {"phase__Fovea5B"}   # byte-identical duplicate of Fovea5A


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def color_of(variant):
    return BASELINE if variant == "baseline" else VARIANT_COLOR[variant]


def load_models():
    loaded = []
    for label, run, ck, method, variant, arch in MODELS:
        path = Path("runs") / run / f"eval_test_{ck}_snr10" / "metrics.json"
        if not path.exists():
            print(f"skip {label}: {path} missing")
            continue
        data = json.loads(path.read_text())
        recs = [r for r in data["per_bscan"] if r["method"] == method and r.get("volume") not in EXCLUDED_VOLUMES]
        if not recs or "volume" not in recs[0]:
            print(f"skip {label}: no per-B-scan records with volume labels (re-evaluate)")
            continue
        spectral = data.get("spectral", {}).get(method, {})
        loaded.append(dict(label=label, variant=variant, arch=arch, records=recs, spectral=spectral))
    return loaded


def mean_ci(model, metric):
    v = [r[metric] for r in model["records"]]
    lo, hi = cluster_bootstrap_ci(v, [r["volume"] for r in model["records"]])
    return float(np.nanmean(v)), lo, hi


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


# Label placement per panel: (dx, dy) in points from the marker, with a leader line. Models
# cluster tightly, so offsets are set by hand to keep every label clear of the others.
LABEL_OFFSETS = {
    "amp_phase": {"Interpolation": (0, -22), "U-Net": (-40, 26), "Cascade": (30, 30),
                  "U-Net no phase terms": (40, -26), "U-Net + D": (-10, -30), "Cascade + D": (-30, 28),
                  "U-Net + power": (0, 22), "U-Net + D + power": (30, -22), "U-Net amplitude-only": (-60, 16)},
    "real_dphi": {"Interpolation": (-20, -24), "U-Net": (0, 26), "Cascade": (-40, 22),
                  "U-Net no phase terms": (-30, -26), "U-Net + D": (-50, 24), "Cascade + D": (-70, 8),
                  "U-Net + power": (20, -26), "U-Net + D + power": (-60, -14), "U-Net amplitude-only": (30, 12)},
}


def point(ax, m, x, y, xerr, yerr, offset):
    c = color_of(m["variant"])
    ax.errorbar(x, y, xerr=[[x - xerr[0]], [xerr[1] - x]], yerr=[[y - yerr[0]], [yerr[1] - y]],
                fmt="none", ecolor=c, elinewidth=1, alpha=0.45, zorder=2)
    ax.scatter([x], [y], s=64, marker=ARCH_MARKER[m["arch"]], color=c, edgecolors=SURFACE,
               linewidths=2, zorder=3)
    ax.annotate(m["label"], (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                color=INK_2, ha="center", va="center", zorder=4,
                arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.6, shrinkA=2, shrinkB=5))


def fig_tradeoff(models, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [("psnr_db_tissue", "PSNR in tissue [dB] →", "phase_err_w_rad", "← phase error, amplitude-weighted [rad]",
               "Amplitude fidelity vs phase fidelity", "amp_phase"),
              ("hist_sim", "amplitude histogram similarity (HistSim) →", "dphase_err_rad",
               "← inter-A-line Δφ error [rad]", "Speckle realism vs Doppler-phase fidelity", "real_dphi")]
    for ax, (xm, xl, ym, yl, title, key) in zip(axes, panels):
        for m in models:
            x, xlo, xhi = mean_ci(m, xm)
            y, ylo, yhi = mean_ci(m, ym)
            point(ax, m, x, y, (xlo, xhi), (ylo, yhi), LABEL_OFFSETS[key].get(m["label"], (0, 20)))
        ax.margins(x=0.12, y=0.12)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        ax.invert_yaxis()   # up = better phase
    handles = [plt.Line2D([], [], marker=ARCH_MARKER[a], linestyle="", color=INK_2, markersize=7, label=l)
               for a, l in (("none", "no network"), ("unet", "U-Net"), ("cascade", "DC cascade"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Test set (5 volumes): means with 95% CIs over volumes. Up and right is better.",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out, "tradeoff")


def fig_boxplots(models, out):
    metrics = [("psnr_db_tissue", "PSNR in tissue [dB] ↑"), ("hist_sim", "HistSim ↑"),
               ("unmeasured_power_ratio", "missing-line power ratio (GT ≈ 1) →1"),
               ("phase_err_w_rad", "phase error [rad] ↓"), ("dphase_err_rad", "Δφ error [rad] ↓"),
               ("wpc", "WPC ↑")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    labels = [m["label"] for m in models]
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [[r[key] for r in m["records"]] for m in models]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK_2, linewidth=1),
                        capprops=dict(color=INK_2, linewidth=1))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(color_of(m["variant"]))
            patch.set_alpha(0.55)
            patch.set_edgecolor(color_of(m["variant"]))
        if key == "unmeasured_power_ratio":
            gt = np.nanmean([r["unmeasured_power_ratio_gt"] for r in models[0]["records"]])
            ax.axhline(gt, color=INK_2, linewidth=1)
            ax.annotate("ground truth", (len(models) + 0.4, gt), fontsize=7, color=INK_2, va="bottom", ha="right")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in axes[-1]:
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=40, ha="right")
    fig.suptitle("Per-B-scan distributions on the test set (160 B-scans; box = quartiles, whiskers = 1.5 IQR)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "boxplots")


def fig_deciles(models, out):
    keep = [m for m in models if m["label"] in
            ("Interpolation", "U-Net", "U-Net + D", "U-Net + power", "U-Net amplitude-only", "U-Net + D + power")]
    x = np.arange(1, 11)
    curves = {m["label"]: np.array([np.nanmean([r[f"coh_d{i}"] for r in m["records"]]) for i in x]) for m in keep}
    base = curves["Interpolation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, relative in zip(axes, (False, True)):
        ends = []
        for m in keep:
            y = curves[m["label"]] - base if relative else curves[m["label"]]
            c = color_of(m["variant"])
            ax.plot(x, y, color=c, linewidth=2, zorder=2)
            ax.scatter([x[-1]], [y[-1]], s=40, color=c, edgecolors=SURFACE, linewidths=2, zorder=3,
                       marker=ARCH_MARKER[m["arch"]])
            ends.append((y[-1], m["label"]))
        ax.set_xlim(0.7, 10.3)
        ax.set_xticks(x, [f"d{i}" for i in x])
        ax.set_xlabel("ground-truth amplitude decile (d1 = weakest; d10 = strongest)")
        if relative:
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.set_ylabel("gain over interpolation in mean cos(φ̂ − φ)")
            ax.set_title("Gain over interpolation", loc="left")
        else:
            ax.set_ylabel("mean cos(φ̂ − φ)")
            ax.set_title("Phase coherence by signal level", loc="left")
    # Curves converge at the right edge, so identity goes in a legend (ordered by the d10 value).
    order = sorted(keep, key=lambda m: -curves[m["label"]][-1])
    handles = [plt.Line2D([], [], color=color_of(m["variant"]), marker=ARCH_MARKER[m["arch"]], markersize=6,
                          label=m["label"]) for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "deciles")


# ---------------------------------------------------------------- qualitative figures

def _db(z):
    return 20 * np.log10(np.abs(z) + 1e-12)


def _noise_floor(z):
    row = np.median(np.abs(z), axis=1)
    k = max(1, len(row) // 10)
    return np.median(np.sort(row)[:k])


def _zoom_boxes(gt, size=96, n=2, snr_db=10.0):
    """n non-overlapping size×size windows with the most above-noise energy."""
    amp = np.abs(gt)
    e = np.where(amp > _noise_floor(gt) * 10 ** (snr_db / 20), amp ** 2, 0)
    from scipy.ndimage import uniform_filter
    score = uniform_filter(e, size)
    boxes = []
    for _ in range(n):
        z, x = np.unravel_index(np.argmax(score), score.shape)
        z0, x0 = int(np.clip(z - size // 2, 0, gt.shape[0] - size)), int(np.clip(x - size // 2, 0, gt.shape[1] - size))
        boxes.append((z0, x0))
        score[max(0, z0 - size):z0 + 2 * size, max(0, x0 - size):x0 + 2 * size] = -1
    return boxes, size


def fig_qualitative(npz_path: Path, out: Path, snr_db: float = 10.0):
    d = np.load(npz_path, allow_pickle=True)
    labels = [str(s) for s in d["labels"]]
    fields = {lab: d[f"f{i}"] for i, lab in enumerate(labels)}
    sample = str(d["sample"])
    gt = fields["Ground truth"]
    boxes, size = _zoom_boxes(gt)
    vmin, vmax = -45, 0
    box_colors = ("#2a78d6", "#eb6834")

    # Amplitude: rows = methods; columns = full B-scan + zooms.
    n = len(labels)
    fig, axes = plt.subplots(n, 1 + len(boxes), figsize=(3.2 * (1 + len(boxes)), 2.6 * n),
                             gridspec_kw=dict(width_ratios=[1.6] + [1] * len(boxes)), squeeze=False)
    for r, lab in enumerate(labels):
        db = _db(fields[lab])
        ax = axes[r, 0]
        ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        for (z0, x0), bc in zip(boxes, box_colors):
            ax.add_patch(Rectangle((x0, z0), size, size, fill=False, ec=bc, lw=1.4))
        ax.set_ylabel(lab, fontsize=9, color=INK)
        for c, ((z0, x0), bc) in enumerate(zip(boxes, box_colors), start=1):
            a = axes[r, c]
            a.imshow(db[z0:z0 + size, x0:x0 + size], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            for s in a.spines.values():
                s.set_visible(True)
                s.set_edgecolor(bc)
                s.set_linewidth(1.6)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
    axes[0, 0].set_title("amplitude [dB]", loc="left")
    for c in range(1, 1 + len(boxes)):
        axes[0, c].set_title(f"zoom {c}", loc="left")
    fig.suptitle(f"{sample}: {str(d['volume']).split('__')[-1]}, B-scan {int(d['y'])}, K={int(d['factor'])} "
                 f"(dB relative to the volume's 99.9th-percentile amplitude)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out, f"qual_amplitude_{sample}")

    # Phase continuity: rows = axial / lateral phase difference; columns = methods (tissue only).
    amp = np.abs(gt)
    mask = amp > _noise_floor(gt) * 10 ** (snr_db / 20)
    (z0, x0) = boxes[0]
    sl = (slice(z0, z0 + size), slice(x0, x0 + size))
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0), squeeze=False)
    for c, lab in enumerate(labels):
        f = fields[lab][sl]
        m = mask[sl]
        dz = np.angle(f[1:] * f[:-1].conj())
        dx = np.angle(f[:, 1:] * f[:, :-1].conj())
        for r, (img, mm) in enumerate(((dz, m[1:] & m[:-1]), (dx, m[:, 1:] & m[:, :-1]))):
            ax = axes[r, c]
            im = ax.imshow(np.where(mm, img, np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi,
                           interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_facecolor("#f0efec")
        axes[0, c].set_title(lab, fontsize=8.5, loc="left")
    axes[0, 0].set_ylabel("axial Δφ_z", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("lateral Δφ_x", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("phase difference [rad]", color=INK_2)
    fig.suptitle(f"{sample}: phase continuity in zoom 1 (pixels ≥ {snr_db:g} dB above the noise floor; grey = masked)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    save(fig, out, f"qual_phase_{sample}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/figures")
    p.add_argument("--only", nargs="*", default=["tradeoff", "boxplots", "deciles", "qual"])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    if {"tradeoff", "boxplots", "deciles"} & set(args.only):
        models = load_models()
        if "tradeoff" in args.only:
            fig_tradeoff(models, out)
        if "boxplots" in args.only:
            fig_boxplots(models, out)
        if "deciles" in args.only:
            fig_deciles(models, out)
    if "qual" in args.only:
        files = sorted(out.glob("recon_*.npz"))
        if not files:
            print(f"no {out}/recon_*.npz yet (run scripts/dump_reconstructions.py on the cluster)")
        for f in files:
            fig_qualitative(f, out)


if __name__ == "__main__":
    main()
