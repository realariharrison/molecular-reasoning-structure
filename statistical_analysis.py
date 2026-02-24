"""
Statistical Analysis

Tests pre-registered hypotheses H1-H5 with proper multiple comparison correction.

H1: Structural stability correlates with prediction accuracy (point-biserial r > 0.15)
H2: Correlation stronger for hard endpoints (hERG, P-gp) than easy (CYP2D6)
H3: Higher covalent ratio -> higher accuracy; higher vdw -> lower accuracy
H4: Incorrect predictions have lower structural stability (two-sample t-test)
H5: CYP2D6 reasoning shows more covalent; hERG shows more vdw bonds
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Multiple comparison correction
N_TESTS = 30  # Approximate total tests
BONFERRONI_ALPHA = 0.05 / N_TESTS


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""
    hypothesis: str
    description: str
    statistic_name: str
    statistic_value: float
    p_value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    n: int = 0
    significant_bonferroni: bool = False
    significant_bh: bool = False  # Set after BH correction

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert numpy types to Python native for JSON serialization
        for k, v in d.items():
            if isinstance(v, (np.bool_, np.integer)):
                d[k] = v.item()
            elif isinstance(v, np.floating):
                d[k] = float(v)
        return d


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_func, n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic between x and y."""
    rng = np.random.RandomState(42)
    stats_boot = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            s = stat_func(x[idx], y[idx])
            if not np.isnan(s):
                stats_boot.append(s)
        except (ValueError, ZeroDivisionError):
            continue
    if not stats_boot:
        return (np.nan, np.nan)
    return (np.percentile(stats_boot, 100 * alpha / 2),
            np.percentile(stats_boot, 100 * (1 - alpha / 2)))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of significant flags."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    thresholds = [(i + 1) / n * alpha for i in range(n)]
    significant = [False] * n

    # Find largest k where p(k) <= k/m * alpha
    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k

    if max_k >= 0:
        for k in range(max_k + 1):
            significant[sorted_indices[k]] = True

    return significant


def test_h1(df: pd.DataFrame) -> HypothesisResult:
    """H1: Structural stability positively correlates with accuracy."""
    valid = df[df["structural_stability"].notna() & df["is_correct"].notna()].copy()

    stability = valid["structural_stability"].values
    correct = valid["is_correct"].astype(float).values

    r, p = stats.pointbiserialr(correct, stability)

    ci_low, ci_high = bootstrap_ci(
        correct, stability,
        lambda x, y: stats.pointbiserialr(x, y)[0]
    )

    return HypothesisResult(
        hypothesis="H1",
        description="Structural stability correlates with prediction accuracy",
        statistic_name="point_biserial_r",
        statistic_value=round(r, 4),
        p_value=p,
        ci_lower=round(ci_low, 4) if not np.isnan(ci_low) else None,
        ci_upper=round(ci_high, 4) if not np.isnan(ci_high) else None,
        effect_size=round(r, 4),
        effect_size_name="r",
        n=len(valid),
        significant_bonferroni=p < BONFERRONI_ALPHA,
    )


def test_h2(df: pd.DataFrame) -> list[HypothesisResult]:
    """H2: Correlation stronger for hard endpoints than easy endpoints."""
    results = []
    hard_endpoints = ["herg", "pgp_broccatelli"]
    easy_endpoints = ["cyp2d6_veith"]

    for endpoint in df["endpoint"].unique():
        subset = df[(df["endpoint"] == endpoint) &
                     df["structural_stability"].notna() &
                     df["is_correct"].notna()]
        if len(subset) < 10:
            continue
        r, p = stats.pointbiserialr(
            subset["is_correct"].astype(float).values,
            subset["structural_stability"].values
        )
        difficulty = "hard" if endpoint in hard_endpoints else "easy" if endpoint in easy_endpoints else "medium"
        results.append(HypothesisResult(
            hypothesis="H2",
            description=f"Stability-accuracy correlation for {endpoint} ({difficulty})",
            statistic_name="point_biserial_r",
            statistic_value=round(r, 4),
            p_value=p,
            n=len(subset),
            significant_bonferroni=p < BONFERRONI_ALPHA,
        ))

    return results


def test_h3(df: pd.DataFrame) -> list[HypothesisResult]:
    """H3: Bond type ratios predict accuracy direction."""
    results = []
    valid = df[df["is_correct"].notna()].copy()

    for ratio_col, expected_direction in [
        ("covalent_ratio", "positive"),
        ("vdw_ratio", "negative"),
        ("hydrogen_ratio", "non-monotonic"),
    ]:
        subset = valid[valid[ratio_col].notna()]
        if len(subset) < 10:
            continue

        r, p = stats.pointbiserialr(
            subset["is_correct"].astype(float).values,
            subset[ratio_col].values
        )

        results.append(HypothesisResult(
            hypothesis="H3",
            description=f"{ratio_col} vs accuracy (expected: {expected_direction})",
            statistic_name="point_biserial_r",
            statistic_value=round(r, 4),
            p_value=p,
            n=len(subset),
            significant_bonferroni=p < BONFERRONI_ALPHA,
        ))

    return results


def test_h4(df: pd.DataFrame) -> HypothesisResult:
    """H4: Incorrect predictions have lower structural stability."""
    valid = df[df["structural_stability"].notna() & df["is_correct"].notna()]

    correct = valid[valid["is_correct"] == True]["structural_stability"].values
    incorrect = valid[valid["is_correct"] == False]["structural_stability"].values

    if len(correct) < 5 or len(incorrect) < 5:
        return HypothesisResult(
            hypothesis="H4",
            description="Incorrect predictions have lower stability",
            statistic_name="t_statistic",
            statistic_value=0.0,
            p_value=1.0,
            n=len(valid),
        )

    t, p = stats.ttest_ind(correct, incorrect, alternative="greater")
    d = cohens_d(correct, incorrect)

    return HypothesisResult(
        hypothesis="H4",
        description="Incorrect predictions have lower structural stability",
        statistic_name="t_statistic",
        statistic_value=round(t, 4),
        p_value=p,
        effect_size=round(d, 4),
        effect_size_name="cohens_d",
        n=len(valid),
        significant_bonferroni=p < BONFERRONI_ALPHA,
    )


def test_h5(df: pd.DataFrame) -> list[HypothesisResult]:
    """H5: CYP2D6 has more covalent bonds; hERG has more vdw bonds."""
    results = []

    cyp = df[df["endpoint"] == "cyp2d6_veith"]
    herg = df[df["endpoint"] == "herg"]

    if len(cyp) >= 10 and len(herg) >= 10:
        # CYP2D6 vs hERG covalent ratio
        t, p = stats.ttest_ind(
            cyp["covalent_ratio"].dropna().values,
            herg["covalent_ratio"].dropna().values,
            alternative="greater"
        )
        results.append(HypothesisResult(
            hypothesis="H5",
            description="CYP2D6 has higher covalent ratio than hERG",
            statistic_name="t_statistic",
            statistic_value=round(t, 4),
            p_value=p,
            effect_size=round(cohens_d(
                cyp["covalent_ratio"].dropna().values,
                herg["covalent_ratio"].dropna().values
            ), 4),
            effect_size_name="cohens_d",
            n=len(cyp) + len(herg),
            significant_bonferroni=p < BONFERRONI_ALPHA,
        ))

        # hERG vs CYP2D6 vdw ratio
        t, p = stats.ttest_ind(
            herg["vdw_ratio"].dropna().values,
            cyp["vdw_ratio"].dropna().values,
            alternative="greater"
        )
        results.append(HypothesisResult(
            hypothesis="H5",
            description="hERG has higher vdw ratio than CYP2D6",
            statistic_name="t_statistic",
            statistic_value=round(t, 4),
            p_value=p,
            effect_size=round(cohens_d(
                herg["vdw_ratio"].dropna().values,
                cyp["vdw_ratio"].dropna().values
            ), 4),
            effect_size_name="cohens_d",
            n=len(cyp) + len(herg),
            significant_bonferroni=p < BONFERRONI_ALPHA,
        ))

    return results


def logistic_model(df: pd.DataFrame) -> dict:
    """Supplementary: logistic regression with reasoning structure features."""
    valid = df.dropna(subset=["is_correct", "covalent_ratio", "hydrogen_ratio", "vdw_ratio", "structural_stability"])
    if len(valid) < 20:
        return {"error": "insufficient data"}

    X = valid[["covalent_ratio", "hydrogen_ratio", "vdw_ratio", "structural_stability"]].values
    y = valid["is_correct"].astype(int).values

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, y_prob)

    return {
        "auroc": round(auroc, 4),
        "coefficients": {
            "covalent_ratio": round(model.coef_[0][0], 4),
            "hydrogen_ratio": round(model.coef_[0][1], 4),
            "vdw_ratio": round(model.coef_[0][2], 4),
            "structural_stability": round(model.coef_[0][3], 4),
        },
        "intercept": round(model.intercept_[0], 4),
        "n": len(valid),
    }


def run_all_tests(
    bond_summary_path: Path = DATA_DIR / "bonds" / "bond_summary.csv",
    stability_path: Path = DATA_DIR / "stability_scores.csv",
    output_path: Path = DATA_DIR / "hypothesis_results.json",
) -> dict:
    """Run all hypothesis tests and save results."""

    # Load and merge data
    bonds_df = pd.read_csv(bond_summary_path)
    stability_df = pd.read_csv(stability_path)

    df = bonds_df.merge(
        stability_df[["trace_id", "structural_stability"]],
        on="trace_id",
        how="left",
        suffixes=("", "_stab"),
    )

    logger.info(f"Loaded {len(df)} traces for analysis")

    # Run tests
    all_results = []

    logger.info("\nH1: Stability-accuracy correlation...")
    h1 = test_h1(df)
    all_results.append(h1)
    logger.info(f"  r={h1.statistic_value}, p={h1.p_value:.6f}, n={h1.n}")

    logger.info("\nH2: Per-endpoint correlations...")
    h2_results = test_h2(df)
    all_results.extend(h2_results)
    for r in h2_results:
        logger.info(f"  {r.description}: r={r.statistic_value}, p={r.p_value:.6f}")

    logger.info("\nH3: Bond ratios vs accuracy...")
    h3_results = test_h3(df)
    all_results.extend(h3_results)
    for r in h3_results:
        logger.info(f"  {r.description}: r={r.statistic_value}, p={r.p_value:.6f}")

    logger.info("\nH4: Stability difference (correct vs incorrect)...")
    h4 = test_h4(df)
    all_results.append(h4)
    logger.info(f"  t={h4.statistic_value}, p={h4.p_value:.6f}, d={h4.effect_size}")

    logger.info("\nH5: Endpoint-specific bond patterns...")
    h5_results = test_h5(df)
    all_results.extend(h5_results)
    for r in h5_results:
        logger.info(f"  {r.description}: t={r.statistic_value}, p={r.p_value:.6f}")

    # Benjamini-Hochberg correction (replace NaN p-values with 1.0)
    p_values = [r.p_value if not np.isnan(r.p_value) else 1.0 for r in all_results]
    bh_significant = benjamini_hochberg(p_values)
    for r, sig in zip(all_results, bh_significant):
        r.significant_bh = sig

    # Logistic model
    logger.info("\nLogistic regression model...")
    logistic = logistic_model(df)
    logger.info(f"  AUROC: {logistic.get('auroc', 'N/A')}")

    # Save
    output = {
        "n_traces": len(df),
        "n_tests": len(all_results),
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "hypotheses": [r.to_dict() for r in all_results],
        "logistic_model": logistic,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Summary
    n_sig_bonf = sum(1 for r in all_results if r.significant_bonferroni)
    n_sig_bh = sum(1 for r in all_results if r.significant_bh)
    logger.info(f"\nSignificant (Bonferroni): {n_sig_bonf}/{len(all_results)}")
    logger.info(f"Significant (BH FDR):    {n_sig_bh}/{len(all_results)}")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_all_tests()
