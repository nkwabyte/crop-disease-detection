"""Figure 4: iOS resident memory by benchmark phase (iPhone 15 Pro Max, Release build, 2026-11-20).

Reads only archived evidence:
  paper_evidence/mobile/iphone_15_pro_max_release_2026-11-20/console_phase_trace.txt
      '[iOS-Bench] phase=<name> residentMB=<v>' lines printed by the app at each
      phase boundary (v = mach resident_size / 1024^2, i.e. MiB despite the label),
      including 'after_end_to_end'. (Redrawn 2026-11-20; the 2026-11-19 debug-build
      version took its last point from the export because that console line was lost.)

The pre-fix build left no trace; its termination at about 1,600 MiB (session record,
2026-09-14) is drawn as an annotated reference line, not as data points.

Usage: .venv-eval/bin/python scripts/revision/make_F04_memory.py OUT_DIR [OUT_DIR ...]
"""
import json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "paper_evidence" / "mobile" / "iphone_15_pro_max_release_2026-11-20"

LABELS = {
    "start": "Start",
    "after_cold_load_loop": "Cold-load\nloop",
    "after_artifacts_and_images": "Models,\nimages",
    "after_classifier_stage": "Classifier\nstage",
    "detector_stage_1920x1080_done": "Detector\n1920×1080",
    "detector_stage_1280x960_done": "Detector\n1280×960",
    "detector_stage_640x480_done": "Detector\n640×480",
    "after_end_to_end": "After\nend-to-end\nloop",
}


def main(outs):
    trace = {}
    for line in (EV / "console_phase_trace.txt").read_text().splitlines():
        m = re.search(r"phase=(\S+) residentMB=([\d.]+)", line)
        if m:
            trace[m.group(1)] = float(m.group(2))
    xs, ys = [], []
    for k, lab in LABELS.items():
        xs.append(lab); ys.append(trace[k])

    green, orange = "#009E73", "#D55E00"
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
    ax.plot(range(len(xs)), ys, "-o", color=green, lw=2.4, ms=6,
            label="Contiguous-buffer build (Release, recorded 2026-11-20)")
    for i, y in enumerate(ys):
        ax.annotate(f"{y:.1f}", (i, y), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8.5, color=green)
    ax.axhline(1600, color=orange, ls="--", lw=1.6)
    ax.text(0, 1650, "Per-value object build: terminated by iOS at about 1,600 MiB\n"
            "(session record, 2026-09-14; no trace survives)",
            color=orange, fontsize=8.5, va="bottom", ha="left")
    ax.set_yscale("log")
    ticks = [25, 50, 100, 200, 400, 800, 1600]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(25, 3200)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8.5)
    ax.set_ylabel("Resident memory (MiB, logarithmic scale)")
    ax.set_xlabel("Benchmark phase boundary")
    ax.grid(axis="y", ls=":", color="0.7")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.62), frameon=False, fontsize=8.5)
    fig.tight_layout()
    for o in outs:
        o = Path(o); o.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(o / f"F04_memory_pre_post_fix.{ext}")
    print(dict(zip([x.replace("\n", " ") for x in xs], [round(y, 1) for y in ys])))


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
