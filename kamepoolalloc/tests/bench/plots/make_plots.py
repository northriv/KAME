#!/usr/bin/env python3
"""Regenerate the README benchmark SVGs from the tables in
kamepoolalloc/README.md.  Data is inlined below — when the README tables
are re-measured, update the arrays here and re-run:

    python3 make_plots.py        # writes ../../../doc/bench/*.svg

Requires matplotlib (any recent version).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "doc", "bench")
os.makedirs(OUT, exist_ok=True)

# Shared cosmetics — colors readable on both light and dark GitHub themes.
C_KAME = "#d62728"   # red, bold
C_SYS  = "#7f7f7f"   # gray
C_MI   = "#1f77b4"   # blue
C_JE   = "#2ca02c"   # green
LW_KAME, LW_OTHER = 3.0, 1.8

SIZES7 = ["64 B", "1 KiB", "16 KiB", "64 KiB", "256 KiB", "1 MiB", "4 MiB"]
X7 = range(7)

def style(ax, title, ylabel="M ops/s (log)"):
    ax.set_title(title, fontsize=11)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

def sweep(fname, title, sys_, mi, je, kame, note=None, note_xy=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=100)
    ax.plot(X7, sys_, "o-", color=C_SYS, lw=LW_OTHER, ms=4, label="system")
    ax.plot(X7, mi,  "s-", color=C_MI, lw=LW_OTHER, ms=4, label="mimalloc")
    ax.plot(X7, je,  "^-", color=C_JE, lw=LW_OTHER, ms=4, label="jemalloc")
    ax.plot(X7, kame, "o-", color=C_KAME, lw=LW_KAME, ms=6, label="kame",
            zorder=5)
    ax.set_xticks(list(X7), SIZES7)
    style(ax, title)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    if note:
        ax.annotate(note, xy=note_xy, fontsize=8.5, color=C_MI,
                    ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)

# ── bench_loop single-thread sweeps ────────────────────────────────────
# Apple M3 (arm64, macOS), kame @ 60971013, median of 7.
sweep("bench_loop_m3_1t.svg",
      "bench_loop, 1 thread — Apple M3 (macOS arm64)",
      sys_=[106, 86, 91, 25, 25, 25, 45],
      mi=[503, 447, 198, 199, 202, 7, 6],
      je=[128, 124, 72, 18, 18, 18, 18],
      kame=[651, 428, 346, 142, 131, 121, 112],
      note="mmap-per-call cliff", note_xy=(5.0, 10.5))

# Ohtaka (AMD EPYC 7702, Linux bare metal), kame @ efbe6dcb, median of 5.
sweep("bench_loop_ohtaka_1t.svg",
      "bench_loop, 1 thread — AMD EPYC 7702 (Linux)",
      sys_=[212, 214, 69, 65, 70, 71, 67],
      mi=[331, 203, 93, 95, 95, 5, 4],
      je=[182, 160, 45, 4, 4, 4, 4],
      kame=[260, 236, 168, 82, 87, 84, 62],
      note="mmap-per-call cliff", note_xy=(4.8, 7.5))

# ── Ohtaka 128-process aggregate (grouped bars) ────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=100)
sizes4 = ["64 B", "16 KiB", "64 KiB", "1 MiB"]
data = {  # G ops/s aggregate
    "system":   ([22634, 7920, 8128, 8908], C_SYS),
    "mimalloc": ([39572, 11808, 11866, 566], C_MI),
    "jemalloc": ([21280, 5282, 465, 475], C_JE),
    "kame":     ([29044, 21496, 10352, 10703], C_KAME),
}
w = 0.2
for i, (label, (vals, color)) in enumerate(data.items()):
    xs = [x + (i - 1.5) * w for x in range(4)]
    ax.bar(xs, [v / 1000.0 for v in vals], width=w, color=color, label=label)
ax.set_xticks(range(4), sizes4)
style(ax, "bench_loop, 128 parallel processes — EPYC 7702", ylabel="G ops/s (log)")
ax.legend(frameon=False, fontsize=9, ncols=4, loc="upper right")
ax.annotate("19×", xy=(3 + 0.5 * w, 10.7), xytext=(3 + 0.5 * w, 20),
            fontsize=10, color=C_KAME, ha="center", weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_loop_ohtaka_128p.svg"))
plt.close(fig)

# ── thread scaling (alloc_tune_report, Ohtaka) ─────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=100)
threads = [1, 4, 16, 64, 128]
ax.plot(threads, [t * 121 for t in threads], "--", color="#bbbbbb", lw=1.2,
        label="linear (×121)")
ax.plot(threads, [121, 482, 1887, 6356, 10596], "o-", color=C_KAME,
        lw=LW_KAME, ms=6, label="64 B bucket")
ax.plot(threads, [95, 389, 1543, 5713, 7831], "s-", color="#ff7f0e",
        lw=2.2, ms=5, label="1 KiB bucket")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(threads))
ax.set_xticks(threads, [str(t) for t in threads])
ax.set_xlabel("threads")
style(ax, "kame thread scaling — EPYC 7702, 8 NUMA nodes", ylabel="M ops/s aggregate (log)")
ax.annotate("10.6 G ops/s\n(88× linear)", xy=(128, 10596),
            xytext=(40, 11500), fontsize=9, color=C_KAME, weight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "scaling_ohtaka.svg"))
plt.close(fig)

print("wrote SVGs to", os.path.normpath(OUT))
