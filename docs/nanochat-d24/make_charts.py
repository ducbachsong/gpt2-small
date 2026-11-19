"""make_charts.py — the nanochat-d24 model card's charts from the run's train logs (steps 1..3000 from the A/B run's
ab/train.log, 3001..10420 from full/train.log): loss over the steps, val loss against training FLOPs and against
training time, and MFU over the steps. FLOPs a token as tests/tune_model.cu counts them (6 x params + 12 x embed_dim x
the keys a token sees, each layer: nanochat's estimate_flops); MFU against the H100 SXM's bfloat16 tensor-core peak at
its clock (132 SMs x 1,980 MHz x 4,096 = 1,070.5 TFLOPS, tune_helpers.h's peak_bf16_flops). Training time is the sum of
the steps' own times (the val and checkpoint pauses left out). Usage: make_charts.py <log folder> <out folder>"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# the model and the machine
PARAMS = 733_105_152
LAYERS, EMBED, SEQ, WINDOW = 24, 1536, 2048, 512
TOKENS_PER_STEP = 524_288
GPUS = 8
PEAK_BF16 = 132 * 1.98e9 * 4096                    # one H100 SXM, FLOP/s
attention = sum(12 * EMBED * (SEQ if (l % 4 == 3 or l == LAYERS - 1) else WINDOW) for l in range(LAYERS))
FLOPS_PER_TOKEN = 6 * PARAMS + attention            # 4.795e9

# the palette (dataviz reference instance, light): slot 1 blue, slot 2 orange; text and grid inks
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})

STEP_LINE = re.compile(r"^step +(\d+)/\d+ \| loss ([\d.]+) \| norm +([\d.]+) \| lr \S+ \| +(\d+) ms .*\| +(\d+) tokens/s")
VAL_LINE = re.compile(r"^step +(\d+) \| val loss ([\d.]+)")


def read(path, first, last):
    """Gives the log's step lines {step: (loss, norm, ms, tokens/s)} and val lines {step: loss} for first..last."""
    steps, vals = {}, {}
    for line in open(path, encoding="utf-8"):
        if m := STEP_LINE.match(line):
            s = int(m.group(1))
            if first <= s <= last:
                steps[s] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        elif m := VAL_LINE.match(line):
            s = int(m.group(1))
            if first <= s <= last:
                vals[s] = float(m.group(2))
    return steps, vals


steps, vals = read(os.path.join(LOGS, "ab", "train.log"), 0, 3000)
more_steps, more_vals = read(os.path.join(LOGS, "full", "train.log"), 3001, 10_420)
steps.update(more_steps)
vals.update(more_vals)
s = np.array(sorted(steps))
loss = np.array([steps[k][0] for k in s])
ms = np.array([steps[k][2] for k in s])
tps = np.array([steps[k][3] for k in s])
v_steps = np.array(sorted(vals))
v_loss = np.array([vals[k] for k in v_steps])
assert len(s) == 10_420, len(s)
hours = np.cumsum(ms) / 3.6e6                        # training time after each step
flops = s.astype(np.float64) * TOKENS_PER_STEP * FLOPS_PER_TOKEN   # training FLOPs after each step (float: int32
v_hours = np.interp(v_steps, s, hours, left=0.0)                     # numpy ints on Windows overflow here)
v_flops = v_steps.astype(np.float64) * TOKENS_PER_STEP * FLOPS_PER_TOKEN
mfu = tps * FLOPS_PER_TOKEN / (GPUS * PEAK_BF16) * 100


def rolling(values, window):
    """Gives the mean of the last `window` values at each point (fewer at the start)."""
    sums = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return (sums[1:] - sums[np.maximum(np.arange(1, len(values) + 1) - window, 0)]) / counts


def finish(fig, ax, name, note):
    """Adds the source note under the plot and saves the chart as <OUT>/<name>.png."""
    fig.text(0.01, 0.01, note, color=INK_2, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 * (note.count("\n") + 1) + 0.01, 1, 1))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=150)
    plt.close(fig)


final = f"{v_loss[-1]:.3f}"
total_flops = flops[-1]
total_hours = hours[-1]

# 1. loss over the steps: train (mean of 50 steps) and val
fig, ax = plt.subplots(figsize=(9, 5))
train_smooth = rolling(loss, 50)
ax.plot(s, train_smooth, color=BLUE, linewidth=2, label="train loss (mean of 50 steps)")
ax.plot(v_steps, v_loss, color=ORANGE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
        markeredgewidth=1.2, label="val loss")
ax.set_ylim(2.6, 7.0)
ax.set_xlim(0, 10_600)
ax.set_xlabel("step (524,288 tokens each)")
ax.set_ylabel("loss (nats per token)")
ax.set_title("nanochat-d24: loss over training")
ax.annotate(f"val {final}", (v_steps[-1], v_loss[-1]), xytext=(-8, 14), textcoords="offset points",
            ha="right", color=INK, fontsize=10)
ax.legend(loc="upper right", labelcolor=INK_2)
finish(fig, ax, "loss_vs_step", f"8x H100 SXM, 10,420 steps, 5.46B tokens of ClimbMix; the first steps (loss above 7) cut off.")

# 2. val loss against training FLOPs (log scale)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(v_flops[1:], v_loss[1:], color=ORANGE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
        markeredgewidth=1.2)
ax.set_xscale("log")
ax.set_xlabel("training compute (FLOPs, log scale)")
ax.set_ylabel("val loss (nats per token)")
ax.set_title("Val loss against training compute")
mantissa, exponent = f"{total_flops:.2e}".split("e+")
ax.annotate(f"{final} at {mantissa} × 10$^{{{int(exponent)}}}$ FLOPs", (v_flops[-1], v_loss[-1]), xytext=(0, 40),
            textcoords="offset points", ha="right", color=INK, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8))
finish(fig, ax, "loss_vs_flops", f"FLOPs = tokens x {FLOPS_PER_TOKEN / 1e9:.2f}G a token (6 x 733M params + attention, "
       f"nanochat's count); {total_flops / 1e18:.1f} EFLOPs in all.")

# 3. val loss against training time
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(v_hours[1:], v_loss[1:], color=ORANGE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
        markeredgewidth=1.2)
ax.set_xlabel("training time (hours, 8x H100 SXM)")
ax.set_ylabel("val loss (nats per token)")
ax.set_title("Val loss against training time")
ax.set_xlim(0, total_hours * 1.03)
ax.annotate(f"{final} after {total_hours:.2f} h", (v_hours[-1], v_loss[-1]), xytext=(-8, 14),
            textcoords="offset points", ha="right", color=INK, fontsize=10)
finish(fig, ax, "loss_vs_time", "Time: the sum of the training steps' own times (val and checkpoint pauses left out).")

# 4. MFU over the steps (mean of 50 steps)
fig, ax = plt.subplots(figsize=(9, 5))
mfu_smooth = rolling(mfu, 50)
mean_mfu = mfu[10:].mean()
mean_tps = tps[10:].mean()
ax.plot(s, mfu_smooth, color=BLUE, linewidth=2)
ax.axhline(mean_mfu, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
ax.annotate(f"mean {mean_mfu:.1f}%  ({mean_tps:,.0f} tokens/s)", (10_400, mean_mfu), xytext=(0, 8),
            textcoords="offset points", ha="right", color=INK, fontsize=10)
ax.set_ylim(0, 70)
ax.set_xlim(0, 10_600)
ax.set_xlabel("step")
ax.set_ylabel("MFU (% of bfloat16 peak)")
ax.set_title("Model FLOPs utilization over training")
finish(fig, ax, "mfu", f"MFU = tokens/s x {FLOPS_PER_TOKEN / 1e9:.2f}G FLOPs / (8 x 1,070 TFLOPS, H100 SXM bf16 peak at "
       f"1,980 MHz); mean of 50 steps.\nThe products run in FP8, whose peak is twice the bfloat16 peak.")

print(f"steps {len(s)}, val points {len(v_steps)}, FLOPs/token {FLOPS_PER_TOKEN:.4e}, total {total_flops:.3e} FLOPs, "
      f"{total_hours:.3f} h, mean MFU {mean_mfu:.1f}%, mean {mean_tps:,.0f} tokens/s, final val {final}")
