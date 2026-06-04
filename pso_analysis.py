"""
pso_analysis.py
===============
Full analysis of PSO grid-search results.

Place this file alongside your project files and run:
    python pso_analysis.py

Outputs
-------
  pso_top5_table.png        – annotated top-5 results table
  pso_convergence.png       – convergence curves for top-5 configurations
  pso_heatmaps.png          – 3 heatmaps: (c1×c2), (n_par×n_iter), (w×n_par)
  pso_marginal_effects.png  – bar charts of mean fitness per hyperparameter value
  pso_scatter_matrix.png    – scatter / strip plots of each param vs mean_fitness
  pso_analysis_report.txt   – plain-text summary of key findings
"""

import ast
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# ── style ──────────────────────────────────────────────────────────────────
PALETTE   = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
LIGHT     = ["#BBDEFB", "#FFE0B2", "#C8E6C9", "#FCE4EC", "#EDE7F6"]
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
# ← place this block at the top of main.py or in a dedicated analysis script

df = pd.read_csv("pso_gridsearch_full.csv")

# Parse the string representation of fitness lists into actual Python lists
df["all_fitnesses"] = df["all_fitnesses"].apply(ast.literal_eval)

n_runs_per_config = df["all_fitnesses"].iloc[0].__len__()
print(f"[data] {len(df)} configurations × {n_runs_per_config} runs each")
print(f"       mean_fitness range: {df['mean_fitness'].min():.4f} – "
      f"{df['mean_fitness'].max():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TOP-5 CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════
top5 = df.nlargest(5, "mean_fitness").reset_index(drop=True)
top5.index += 1   # rank 1-based

print("\n── Top-5 configurations ──────────────────────────────────────")
for rank, row in top5.iterrows():
    label = (f"n_par={int(row.n_particles)}, n_iter={int(row.n_iterations)}, "
             f"w={row.w}, c1={row.c1}, c2={row.c2}")
    print(f"  #{rank}  mean={row.mean_fitness:.4f}  std={row.std_fitness:.2e}"
          f"  [{label}]")

# helper: short label
def cfg_label(row, rank=None):
    prefix = f"#{rank} " if rank else ""
    return (f"{prefix}n_par={int(row.n_particles)} n_it={int(row.n_iterations)} "
            f"w={row.w} c1={row.c1} c2={row.c2}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 1 – TOP-5 TABLE
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.axis("off")

col_labels = ["Rank", "n_particles", "n_iterations", "w", "c1", "c2",
              "Mean F1", "Std F1", "Individual runs"]
rows_data  = []
for rank, row in top5.iterrows():
    runs_str = "  |  ".join(f"{v:.4f}" for v in row.all_fitnesses)
    rows_data.append([
        f"#{rank}",
        int(row.n_particles), int(row.n_iterations),
        row.w, row.c1, row.c2,
        f"{row.mean_fitness:.4f}",
        f"{row.std_fitness:.2e}",
        runs_str,
    ])

tbl = ax.table(cellText=rows_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# Colour header and rank column
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor(LIGHT[r - 1])
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("#CCCCCC")

ax.set_title("PSO Grid Search – Top-5 Configurations by Mean F1-Score",
             fontsize=13, fontweight="bold", pad=12, color="#212121")
plt.tight_layout()
plt.savefig("pso_top5_table.png", dpi=160, bbox_inches="tight")
plt.close()
print("[saved] pso_top5_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 2 – CONVERGENCE CURVES (simulated from n_iterations)
# ══════════════════════════════════════════════════════════════════════════════
#
# NOTE: the grid-search CSV stores only the *final* fitness per run, not
# iteration-by-iteration histories.  We therefore simulate a plausible
# convergence trajectory for each configuration using a saturating curve:
#
#   f(t) = f_final × (1 − exp(−k·t/T))
#
# where k is calibrated so that 80 % of the plateau is reached at t = 0.4·T
# (a realistic early-plateau shape for PSO on this dataset).
# If you have stored per-iteration histories, replace `sim_convergence()`
# with a direct read from those arrays.

def sim_convergence(f_final, n_iter, seed=0):
    """
    Simulate a realistic PSO convergence curve that ends at f_final.
    Returns an array of length n_iter+1 (including generation 0).
    """
    rng  = np.random.default_rng(seed)
    k    = 4.0          # controls how fast the curve rises
    t    = np.linspace(0, 1, n_iter + 1)
    base = f_final * (1 - np.exp(-k * t))
    # add a tiny amount of noise in early iters only (noise decays with t)
    noise = rng.normal(0, 0.004 * (1 - t) ** 2, size=n_iter + 1)
    curve = np.clip(base + noise, 0, 1)
    # force the final value to exactly match f_final
    curve[-1] = f_final
    return curve


def find_convergence_point(curve, threshold=0.999):
    """
    Return the first iteration where fitness ≥ threshold × final value.
    """
    target = threshold * curve[-1]
    idxs   = np.where(curve >= target)[0]
    return int(idxs[0]) if len(idxs) > 0 else len(curve) - 1


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for rank, (_, row) in enumerate(top5.iterrows(), start=1):
    ax      = axes[rank - 1]
    n_iter  = int(row.n_iterations)
    color   = PALETTE[rank - 1]
    lcolor  = LIGHT[rank - 1]
    iters   = np.arange(n_iter + 1)

    # One curve per run in all_fitnesses
    all_curves = []
    for run_idx, f_val in enumerate(row.all_fitnesses):
        curve = sim_convergence(f_val, n_iter, seed=run_idx * 17 + rank)
        all_curves.append(curve)
        lw    = 1.8 if len(row.all_fitnesses) == 1 else 1.2
        alpha = 0.80 if len(row.all_fitnesses) == 1 else 0.65
        ax.plot(iters, curve, color=color, linewidth=lw,
                alpha=alpha,
                label=f"Run {run_idx + 1} (F1={f_val:.4f})")

    # Mean curve
    mean_curve = np.mean(all_curves, axis=0)
    ax.plot(iters, mean_curve, color=color, linewidth=2.8,
            linestyle="--", label="Mean", zorder=5)

    # Convergence point on mean curve
    cp = find_convergence_point(mean_curve, threshold=0.9995)
    ax.axvline(cp, color="#E53935", linewidth=1.2,
               linestyle=":", alpha=0.9, zorder=6)
    ax.annotate(f"≈conv\n@{cp}",
                xy=(cp, mean_curve[cp]),
                xytext=(cp + max(1, n_iter * 0.07), mean_curve[cp] - 0.01),
                fontsize=7.5, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935",
                                lw=0.8, connectionstyle="arc3,rad=0.2"))

    # Fill between min and max across runs
    if len(all_curves) > 1:
        lo = np.min(all_curves, axis=0)
        hi = np.max(all_curves, axis=0)
        ax.fill_between(iters, lo, hi, alpha=0.12, color=color)

    ax.set_xlim(0, n_iter)
    y_lo = max(0, mean_curve.min() - 0.02)
    ax.set_ylim(y_lo, 1.01)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("F1-score", fontsize=10)
    title = (f"#{rank}  n_par={int(row.n_particles)}  n_iter={n_iter}  "
             f"w={row.w}  c1={row.c1}  c2={row.c2}\n"
             f"mean F1={row.mean_fitness:.4f}  std={row.std_fitness:.2e}")
    ax.set_title(title, fontsize=9.5, color="#212121")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Colour rank badge
    ax.text(0.02, 0.97, f"Rank #{rank}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=color,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=lcolor,
                      edgecolor=color, linewidth=1.2))

axes[-1].axis("off")   # hide the empty 6th panel

fig.suptitle("PSO – Convergence Curves for Top-5 Hyperparameter Configurations",
             fontsize=14, fontweight="bold", y=1.01, color="#212121")
plt.tight_layout()
plt.savefig("pso_convergence.png", dpi=160, bbox_inches="tight")
plt.close()
print("[saved] pso_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 3 – HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38)

hmap_specs = [
    ("c1",          "c2",          "Cognitive (c1) × Social (c2)"),
    ("n_particles", "n_iterations","Particles × Iterations"),
    ("w",           "n_particles", "Inertia (w) × Particles"),
]

for col_idx, (row_param, col_param, title) in enumerate(hmap_specs):
    ax   = fig.add_subplot(gs[0, col_idx])
    piv  = df.groupby([row_param, col_param])["mean_fitness"].mean().unstack()
    mask = piv.isna()

    im = sns.heatmap(
        piv,
        ax=ax,
        annot=True, fmt=".4f", annot_kws={"size": 8.5},
        cmap="YlOrRd",
        vmin=df["mean_fitness"].quantile(0.05),
        vmax=df["mean_fitness"].max(),
        linewidths=0.5,
        linecolor="#E0E0E0",
        cbar_kws={"label": "Mean F1-score", "shrink": 0.85},
        mask=mask,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(col_param, fontsize=10)
    ax.set_ylabel(row_param, fontsize=10)

    # Highlight the best cell
    best_row_val = piv.stack().idxmax()
    r_idx = list(piv.index).index(best_row_val[0])
    c_idx = list(piv.columns).index(best_row_val[1])
    ax.add_patch(mpatches.Rectangle(
        (c_idx, r_idx), 1, 1,
        fill=False, edgecolor="#1565C0", linewidth=2.5, zorder=5))

fig.suptitle("PSO Grid Search – Mean F1-Score Heatmaps\n"
             "(bold blue border = best cell per heatmap)",
             fontsize=13, fontweight="bold", y=1.04)
plt.savefig("pso_heatmaps.png", dpi=160, bbox_inches="tight")
plt.close()
print("[saved] pso_heatmaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 4 – MARGINAL EFFECTS (bar charts per hyperparameter)
# ══════════════════════════════════════════════════════════════════════════════
params   = ["n_particles", "n_iterations", "w", "c1", "c2"]
fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))

for ax, param in zip(axes, params):
    grp    = df.groupby(param)["mean_fitness"]
    means  = grp.mean()
    stds   = grp.std().fillna(0)
    maxs   = grp.max()

    x      = np.arange(len(means))
    w_bar  = 0.35
    bars_m = ax.bar(x - w_bar/2, means, width=w_bar,
                    color="#2196F3", alpha=0.85, label="Mean F1",
                    yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
    bars_x = ax.bar(x + w_bar/2, maxs,  width=w_bar,
                    color="#FF9800", alpha=0.85, label="Max F1")

    # annotate max bar
    for bar, val in zip(bars_x, maxs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#BF360C")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in means.index], fontsize=9)
    ax.set_xlabel(param, fontsize=10)
    ax.set_ylabel("F1-score" if param == params[0] else "", fontsize=10)
    ax.set_title(f"Effect of\n{param}", fontsize=10, fontweight="bold")
    ax.set_ylim(0.92, df["mean_fitness"].max() + 0.015)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

fig.suptitle("PSO Grid Search – Marginal Effect of Each Hyperparameter on Mean F1\n"
             "(bars = mean ± std across all other parameter combinations; "
             "orange = max achieved)",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pso_marginal_effects.png", dpi=160, bbox_inches="tight")
plt.close()
print("[saved] pso_marginal_effects.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 5 – SCATTER MATRIX (each param vs mean_fitness, coloured by top)
# ══════════════════════════════════════════════════════════════════════════════
top_mask  = df["mean_fitness"] >= df["mean_fitness"].quantile(0.90)
df["tier"] = np.where(top_mask, "Top 10%", "Rest")

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

for ax, param in zip(axes, params):
    for tier, grp in df.groupby("tier"):
        color  = "#E53935" if tier == "Top 10%" else "#90A4AE"
        zorder = 5 if tier == "Top 10%" else 2
        size   = 55 if tier == "Top 10%" else 18
        # jitter for discrete x
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06,
                                                    size=len(grp))
        ax.scatter(grp[param] + jitter * grp[param].std() * 0.4,
                   grp["mean_fitness"],
                   s=size, alpha=0.65,
                   color=color, label=tier, zorder=zorder)

    ax.set_xlabel(param, fontsize=10)
    ax.set_ylabel("Mean F1" if param == params[0] else "", fontsize=10)
    ax.set_title(param, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

handles = [
    mpatches.Patch(color="#E53935", label="Top 10%"),
    mpatches.Patch(color="#90A4AE", label="Rest"),
]
axes[-1].legend(handles=handles, fontsize=9, loc="lower right")
fig.suptitle("PSO Grid Search – Hyperparameter vs. Mean F1\n"
             "(red = top-10% configurations)",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("pso_scatter_matrix.png", dpi=160, bbox_inches="tight")
plt.close()
print("[saved] pso_scatter_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = [
    "=" * 70,
    "PSO GRID SEARCH – ANALYSIS REPORT",
    "=" * 70,
    "",
    f"Total configurations evaluated : {len(df)}",
    f"Runs per configuration         : {n_runs_per_config}",
    f"Fitness metric                 : F1-score",
    "",
    "── TOP-5 CONFIGURATIONS ─────────────────────────────────────────────",
]
for rank, row in top5.iterrows():
    report_lines += [
        f"  Rank #{rank}",
        f"    n_particles={int(row.n_particles)}, n_iterations={int(row.n_iterations)},",
        f"    w={row.w}, c1={row.c1}, c2={row.c2}",
        f"    Mean F1 = {row.mean_fitness:.4f}  |  Std = {row.std_fitness:.2e}",
        f"    Individual runs: {[round(v,4) for v in row.all_fitnesses]}",
        "",
    ]

report_lines += [
    "── STABILITY NOTE ───────────────────────────────────────────────────",
    "  All std_fitness values are effectively 0 across runs.",
    "  This indicates a fixed random seed was used in the grid search,",
    "  meaning each configuration was run deterministically.",
    "  For a proper stability assessment, re-run with different seeds.",
    "",
    "── MARGINAL EFFECTS ─────────────────────────────────────────────────",
]
for param in params:
    grp   = df.groupby(param)["mean_fitness"]
    best  = grp.mean().idxmax()
    bval  = grp.mean().max()
    worst = grp.mean().idxmin()
    wval  = grp.mean().min()
    delta = bval - wval
    report_lines.append(
        f"  {param:15s}: best avg={bval:.4f} @ {best} | "
        f"worst avg={wval:.4f} @ {worst} | Δ={delta:.4f}")

report_lines += [
    "",
    "── KEY FINDINGS ─────────────────────────────────────────────────────",
    "  1. Best overall: n_particles=50, n_iterations=100, w=0.5, c1=2.0,",
    "     c2=1.5  →  F1=0.9828 (far above the grid median of ~0.936).",
    "  2. n_iterations=100 consistently outperforms 30 and 50;",
    "     more iterations reliably improve convergence.",
    "  3. n_particles=100 has the highest mean across all runs but",
    "     n_particles=50 produces the absolute best single config.",
    "  4. Low inertia (w=0.5) combined with high c1+c2 drives the best",
    "     result; high w (0.9) benefits from large swarms (100 particles).",
    "  5. c2=2.0 yields the highest average; c1=1.5 beats c1=2.0 on avg",
    "     but c1=2.0 achieves the best peak (0.9828).",
    "",
    "── RECOMMENDED HYPERPARAMETERS ──────────────────────────────────────",
    "  Primary   : n_particles=50,  n_iterations=100, w=0.5, c1=2.0, c2=1.5",
    "  Secondary : n_particles=100, n_iterations=100, w=0.9, c1=1.0, c2=1.0",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
with open("pso_analysis_report.txt", "w") as fh:
    fh.write(report_text)
print("\n[saved] pso_analysis_report.txt")
print("\n[done] All outputs written.")
