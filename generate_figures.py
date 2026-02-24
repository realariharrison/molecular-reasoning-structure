"""
Figure Generation

Generates publication-quality figures for the paper.
All figures use actual data — no mock/placeholder data.
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
FIGURES_DIR = DATA_DIR / "figures"

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ENDPOINT_NAMES = {
    "cyp2d6_veith": "CYP2D6",
    "cyp3a4_veith": "CYP3A4",
    "cyp2c9_veith": "CYP2C9",
    "herg": "hERG",
    "pgp_broccatelli": "P-gp",
}

ENDPOINT_COLORS = {
    "cyp2d6_veith": "#2196F3",
    "cyp3a4_veith": "#4CAF50",
    "cyp2c9_veith": "#FF9800",
    "herg": "#F44336",
    "pgp_broccatelli": "#9C27B0",
}


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load bond summary, stability scores, and hypothesis results."""
    bonds_df = pd.read_csv(DATA_DIR / "bonds" / "bond_summary.csv")
    stability_df = pd.read_csv(DATA_DIR / "stability_scores.csv")
    df = bonds_df.merge(
        stability_df[["trace_id", "structural_stability"]],
        on="trace_id", how="left", suffixes=("", "_stab")
    )
    df["endpoint_name"] = df["endpoint"].map(ENDPOINT_NAMES)

    with open(DATA_DIR / "hypothesis_results.json") as f:
        hyp_results = json.load(f)

    return df, hyp_results


def fig1_bond_distributions(df: pd.DataFrame):
    """Figure 1: Bond type distributions across endpoints."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)

    for i, (endpoint, name) in enumerate(ENDPOINT_NAMES.items()):
        subset = df[df["endpoint"] == endpoint]
        if subset.empty:
            continue

        data = pd.DataFrame({
            "Covalent": subset["covalent_ratio"],
            "Hydrogen": subset["hydrogen_ratio"],
            "Van der Waals": subset["vdw_ratio"],
        })

        bp = axes[i].boxplot(
            [data["Covalent"], data["Hydrogen"], data["Van der Waals"]],
            labels=["Cov", "H", "VdW"],
            patch_artist=True,
            widths=0.6,
        )
        colors = ["#1976D2", "#388E3C", "#F57C00"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[i].set_title(name, fontweight="bold")
        if i == 0:
            axes[i].set_ylabel("Bond Ratio")

    fig.suptitle("Bond Type Distributions by ADMET Endpoint", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_bond_distributions.pdf")
    fig.savefig(FIGURES_DIR / "fig1_bond_distributions.png")
    plt.close(fig)
    logger.info("  fig1_bond_distributions saved")


def fig2_stability_accuracy(df: pd.DataFrame):
    """Figure 2: Structural stability vs prediction accuracy."""
    valid = df[df["structural_stability"].notna() & df["is_correct"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # 2A: Violin plot correct vs incorrect
    ax = axes[0]
    correct = valid[valid["is_correct"] == True]["structural_stability"]
    incorrect = valid[valid["is_correct"] == False]["structural_stability"]

    parts = ax.violinplot(
        [correct.values, incorrect.values],
        positions=[1, 2], showmeans=True, showmedians=True
    )
    for pc, color in zip(parts["bodies"], ["#4CAF50", "#F44336"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("Structural Stability (λ)")
    ax.set_title("(A) Stability by Prediction Accuracy")

    # Add p-value annotation
    t, p = stats.ttest_ind(correct, incorrect, alternative="greater")
    ax.annotate(f"p = {p:.4f}", xy=(1.5, ax.get_ylim()[1] * 0.95),
                ha="center", fontsize=9, fontstyle="italic")

    # 2B: Per-endpoint correlation
    ax = axes[1]
    for endpoint, name in ENDPOINT_NAMES.items():
        subset = valid[valid["endpoint"] == endpoint]
        if subset.empty:
            continue
        r, p = stats.pointbiserialr(
            subset["is_correct"].astype(float).values,
            subset["structural_stability"].values
        )
        color = ENDPOINT_COLORS[endpoint]
        ax.bar(name, r, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.annotate(sig, xy=(name, r + 0.01 if r >= 0 else r - 0.02),
                    ha="center", fontsize=8)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Point-Biserial r")
    ax.set_title("(B) Stability-Accuracy Correlation by Endpoint")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_stability_accuracy.pdf")
    fig.savefig(FIGURES_DIR / "fig2_stability_accuracy.png")
    plt.close(fig)
    logger.info("  fig2_stability_accuracy saved")


def fig3_bond_accuracy_heatmap(df: pd.DataFrame):
    """Figure 3: Bond composition heatmap (correct vs incorrect per endpoint)."""
    valid = df[df["is_correct"].notna()]

    endpoints = list(ENDPOINT_NAMES.keys())
    bond_types = ["covalent_ratio", "hydrogen_ratio", "vdw_ratio"]
    labels = ["Covalent", "Hydrogen", "VdW"]

    # Build matrix: rows = endpoint x correctness, cols = bond type
    data_correct = []
    data_incorrect = []
    row_labels = []

    for ep in endpoints:
        name = ENDPOINT_NAMES[ep]
        correct = valid[(valid["endpoint"] == ep) & (valid["is_correct"] == True)]
        incorrect = valid[(valid["endpoint"] == ep) & (valid["is_correct"] == False)]

        if not correct.empty:
            data_correct.append([correct[bt].mean() for bt in bond_types])
        else:
            data_correct.append([0, 0, 0])

        if not incorrect.empty:
            data_incorrect.append([incorrect[bt].mean() for bt in bond_types])
        else:
            data_incorrect.append([0, 0, 0])

        row_labels.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    sns.heatmap(
        pd.DataFrame(data_correct, index=row_labels, columns=labels),
        ax=axes[0], annot=True, fmt=".3f", cmap="YlGnBu",
        vmin=0, vmax=0.7, cbar_kws={"label": "Mean Ratio"}
    )
    axes[0].set_title("Correct Predictions")

    sns.heatmap(
        pd.DataFrame(data_incorrect, index=row_labels, columns=labels),
        ax=axes[1], annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=0.7, cbar_kws={"label": "Mean Ratio"}
    )
    axes[1].set_title("Incorrect Predictions")

    fig.suptitle("Bond Composition: Correct vs Incorrect", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_bond_accuracy_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig3_bond_accuracy_heatmap.png")
    plt.close(fig)
    logger.info("  fig3_bond_accuracy_heatmap saved")


def fig4_entropy_convergence(df: pd.DataFrame):
    """Figure 4: Example entropy convergence curves for correct vs incorrect."""
    from reasoning_trace import load_traces
    from structural_stability import entropy_series

    traces_dir = DATA_DIR / "bonds" / "classified"
    traces = load_traces(traces_dir)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot a few correct and incorrect traces
    n_examples = 5
    correct_traces = [t for t in traces if t.is_correct == True and len(t.bonds) > 5][:n_examples]
    incorrect_traces = [t for t in traces if t.is_correct == False and len(t.bonds) > 5][:n_examples]

    for trace in correct_traces:
        ent = entropy_series(trace)
        if ent:
            ax.plot(range(len(ent)), ent, color="#4CAF50", alpha=0.4, linewidth=1)

    for trace in incorrect_traces:
        ent = entropy_series(trace)
        if ent:
            ax.plot(range(len(ent)), ent, color="#F44336", alpha=0.4, linewidth=1)

    # Legend
    ax.plot([], [], color="#4CAF50", label="Correct", linewidth=2)
    ax.plot([], [], color="#F44336", label="Incorrect", linewidth=2)
    ax.legend()

    ax.set_xlabel("Bond Sequence Index")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Entropy Convergence: Correct vs Incorrect Predictions")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_entropy_convergence.pdf")
    fig.savefig(FIGURES_DIR / "fig4_entropy_convergence.png")
    plt.close(fig)
    logger.info("  fig4_entropy_convergence saved")


def fig5_hypothesis_summary(hyp_results: dict):
    """Figure 5: Summary of hypothesis test results."""
    hypotheses = hyp_results["hypotheses"]

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    p_values = []
    colors = []
    for h in hypotheses:
        labels.append(f"{h['hypothesis']}: {h['description'][:40]}...")
        p_values.append(h["p_value"])
        if h["significant_bh"]:
            colors.append("#4CAF50")
        elif h["significant_bonferroni"]:
            colors.append("#FF9800")
        else:
            colors.append("#9E9E9E")

    y_pos = range(len(labels))
    log_p = [-np.log10(max(p, 1e-20)) for p in p_values]

    ax.barh(y_pos, log_p, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("-log₁₀(p-value)")

    # Significance thresholds
    ax.axvline(-np.log10(0.05), color="orange", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.axvline(-np.log10(BONFERRONI_ALPHA), color="red", linestyle="--", linewidth=0.8,
               label=f"Bonferroni α={BONFERRONI_ALPHA:.4f}")
    ax.legend(loc="lower right")

    ax.set_title("Hypothesis Test Results")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_hypothesis_summary.pdf")
    fig.savefig(FIGURES_DIR / "fig5_hypothesis_summary.png")
    plt.close(fig)
    logger.info("  fig5_hypothesis_summary saved")


BONFERRONI_ALPHA = 0.05 / 30


def fig6_classifier_ablation():
    """Figure 6: Classifier signal ablation results."""
    ablation_path = DATA_DIR / "ablations" / "classifier_signals.json"
    if not ablation_path.exists():
        logger.warning("  Ablation data not found, skipping fig6")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    configs = ["full", "no_similarity", "no_distance", "no_markers", "no_energy"]
    config_labels = ["Full Model", "No Similarity", "No Distance", "No Markers", "No Energy"]

    cov = [data[c]["covalent_ratio"] for c in configs]
    hyd = [data[c]["hydrogen_ratio"] for c in configs]
    vdw = [data[c]["vdw_ratio"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax.bar(x - width, cov, width, label="Covalent", color="#1976D2", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, hyd, width, label="Hydrogen", color="#388E3C", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, vdw, width, label="Van der Waals", color="#F57C00", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Bond Ratio")
    ax.set_title("Classifier Signal Ablation", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 0.75)

    # Add delta annotations for ablated configs
    for i, cfg in enumerate(configs[1:], 1):
        delta_vdw = data[cfg].get("vdw_ratio_delta", 0)
        if abs(delta_vdw) > 0.01:
            sign = "+" if delta_vdw > 0 else ""
            ax.annotate(f"{sign}{delta_vdw:.1%}", xy=(x[i] + width, vdw[i] + 0.01),
                        ha="center", fontsize=7, color="#F57C00", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_classifier_ablation.pdf")
    fig.savefig(FIGURES_DIR / "fig6_classifier_ablation.png")
    plt.close(fig)
    logger.info("  fig6_classifier_ablation saved")


def fig7_openthoughts_validation():
    """Figure 7: Comparison of our classifier vs ByteDance on OpenThoughts."""
    validation_path = DATA_DIR / "classifier_validation.json"
    if not validation_path.exists():
        logger.warning("  Validation data not found, skipping fig7")
        return

    with open(validation_path) as f:
        data = json.load(f)

    our = data["our_distributions"]
    bd = data["bytedance_reported"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 7A: Side-by-side bar comparison
    ax = axes[0]
    labels = ["Covalent", "Hydrogen", "Van der Waals"]
    our_vals = [our["covalent_ratio_mean"], our["hydrogen_ratio_mean"], our["vdw_ratio_mean"]]
    bd_vals = [bd["covalent_ratio_mean"], bd["hydrogen_ratio_mean"], bd["vdw_ratio_mean"]]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, our_vals, width, label="Semantic (Ours)", color="#2196F3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, bd_vals, width, label="Attention (ByteDance)", color="#FF5722", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Ratio")
    ax.set_title("(A) Bond Distributions: Ours vs ByteDance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 0.75)

    # 7B: Scatter of the 3 bond types
    ax = axes[1]
    colors = ["#1976D2", "#388E3C", "#F57C00"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(bd_vals[i], our_vals[i], c=color, s=120, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(label, xy=(bd_vals[i], our_vals[i]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=color)

    ax.plot([0, 0.7], [0, 0.7], "k--", alpha=0.3, linewidth=0.8, label="y = x")
    ax.set_xlabel("ByteDance (Attention-based)")
    ax.set_ylabel("Ours (Semantic)")
    ax.set_title(f"(B) Pearson r = {data['pearson_r']:.3f}")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_openthoughts_validation.pdf")
    fig.savefig(FIGURES_DIR / "fig7_openthoughts_validation.png")
    plt.close(fig)
    logger.info("  fig7_openthoughts_validation saved")


def generate_all_figures():
    """Generate all paper figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df, hyp_results = load_data()

    logger.info("Generating figures...")
    fig1_bond_distributions(df)
    fig2_stability_accuracy(df)
    fig3_bond_accuracy_heatmap(df)
    fig4_entropy_convergence(df)
    fig5_hypothesis_summary(hyp_results)
    fig6_classifier_ablation()
    fig7_openthoughts_validation()

    logger.info(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_all_figures()
