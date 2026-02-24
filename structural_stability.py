"""
Structural Stability Score

Computes entropy convergence rate for reasoning traces.
Shannon entropy of bond-type distribution at each prefix, fit exponential decay.

H(t) = H_0 * exp(-lambda * t)
lambda = structural stability score (higher = more stable reasoning)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from reasoning_trace import ReasoningTrace, Bond, BondType, load_traces, save_traces

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def shannon_entropy(counts: dict[BondType, int]) -> float:
    """Compute Shannon entropy of bond type distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def entropy_series(trace: ReasoningTrace) -> list[float]:
    """
    Compute entropy at each prefix of the bond sequence.

    Bonds are ordered by target_index (when they appear in reasoning).
    At each step t, compute entropy of bond types seen so far.
    """
    if not trace.bonds:
        return []

    # Sort bonds by target step index
    sorted_bonds = sorted(trace.bonds, key=lambda b: b.target_index)

    entropies = []
    counts = {BondType.COVALENT: 0, BondType.HYDROGEN: 0, BondType.VAN_DER_WAALS: 0}

    for bond in sorted_bonds:
        counts[bond.bond_type] += 1
        entropies.append(shannon_entropy(counts))

    return entropies


def exp_decay(t: np.ndarray, h0: float, lam: float) -> np.ndarray:
    """Exponential decay: H(t) = H0 * exp(-lambda * t)"""
    return h0 * np.exp(-lam * t)


def fit_stability(entropies: list[float]) -> float | None:
    """
    Fit exponential decay to entropy series and return lambda (stability score).

    Returns None if fit fails or insufficient data.
    """
    if len(entropies) < 3:
        return None

    t = np.arange(len(entropies), dtype=float)
    h = np.array(entropies)

    # Need some variation to fit
    if h.std() < 1e-6:
        # Constant entropy -> perfectly stable or degenerate
        return 0.0 if h.mean() < 0.1 else None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                exp_decay, t, h,
                p0=[h[0] if h[0] > 0 else 1.0, 0.1],
                bounds=([0, 0], [np.inf, 10]),
                maxfev=5000,
            )
        return float(popt[1])  # lambda
    except (RuntimeError, ValueError):
        # Fallback: linear decay rate
        if len(entropies) >= 2 and entropies[0] > 0:
            slope = (entropies[-1] - entropies[0]) / len(entropies)
            return float(-slope / entropies[0]) if entropies[0] != 0 else None
        return None


def compute_stability_scores(
    traces_dir: Path = DATA_DIR / "bonds" / "classified",
    output_path: Path = DATA_DIR / "stability_scores.csv",
) -> pd.DataFrame:
    """Compute structural stability for all classified traces."""
    traces = load_traces(traces_dir)
    logger.info(f"Computing stability for {len(traces)} traces")

    rows = []
    for trace in traces:
        entropies = entropy_series(trace)
        stability = fit_stability(entropies)

        trace.structural_stability = stability

        row = {
            "trace_id": trace.trace_id,
            "endpoint": trace.molecule.endpoint if trace.molecule else None,
            "is_correct": trace.is_correct,
            "structural_stability": stability,
            "n_bonds": len(trace.bonds),
            "n_entropy_points": len(entropies),
            "initial_entropy": entropies[0] if entropies else None,
            "final_entropy": entropies[-1] if entropies else None,
            "covalent_ratio": trace.covalent_ratio,
            "hydrogen_ratio": trace.hydrogen_ratio,
            "vdw_ratio": trace.vdw_ratio,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    valid = df[df["structural_stability"].notna()]
    logger.info(f"\nStability scores computed: {len(valid)}/{len(df)}")
    logger.info(f"  Mean lambda: {valid['structural_stability'].mean():.4f}")
    logger.info(f"  Std lambda:  {valid['structural_stability'].std():.4f}")

    if valid["is_correct"].notna().any():
        correct = valid[valid["is_correct"] == True]
        incorrect = valid[valid["is_correct"] == False]
        logger.info(f"\n  Correct (n={len(correct)}):   lambda={correct['structural_stability'].mean():.4f}")
        logger.info(f"  Incorrect (n={len(incorrect)}): lambda={incorrect['structural_stability'].mean():.4f}")

    # Save updated traces with stability scores
    save_traces(traces, traces_dir)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    compute_stability_scores()
