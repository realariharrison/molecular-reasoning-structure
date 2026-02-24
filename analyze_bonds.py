from __future__ import annotations
"""
Bond Structure Analysis

Runs SemanticBondClassifier on all collected traces and computes bond distributions.
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from reasoning_trace import ReasoningTrace, BondType, load_traces, save_traces
from bond_classifier import SemanticBondClassifier

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
TRACES_DIR = DATA_DIR / "traces"
BONDS_DIR = DATA_DIR / "bonds"


def analyze_all_traces(
    traces_dir: Path = TRACES_DIR,
    output_dir: Path = BONDS_DIR,
    disabled_signals: list[str] | None = None,
) -> pd.DataFrame:
    """
    Classify bonds in all traces and produce summary statistics.

    Returns DataFrame with one row per trace.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load traces
    traces = load_traces(traces_dir)
    logger.info(f"Loaded {len(traces)} traces")

    if not traces:
        logger.error("No traces found")
        return pd.DataFrame()

    # Initialize classifier
    classifier = SemanticBondClassifier(disabled_signals=disabled_signals)

    # Classify
    classified = classifier.classify_traces(traces)

    # Save classified traces
    save_traces(classified, output_dir / "classified")

    # Build summary DataFrame
    rows = []
    for trace in classified:
        if trace.molecule is None:
            continue
        row = {
            "trace_id": trace.trace_id,
            "smiles": trace.molecule.smiles,
            "endpoint": trace.molecule.endpoint,
            "ground_truth": trace.molecule.ground_truth,
            "claude_prediction": trace.molecule.claude_prediction,
            "claude_confidence": trace.molecule.claude_confidence,
            "is_correct": trace.is_correct,
            "n_reasoning_steps": trace.num_reasoning_steps,
            "n_tool_calls": trace.num_tool_calls,
            "n_total_steps": len(trace.steps),
            "total_bonds": trace.total_bonds,
            "covalent_ratio": trace.covalent_ratio,
            "hydrogen_ratio": trace.hydrogen_ratio,
            "vdw_ratio": trace.vdw_ratio,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save
    df.to_csv(output_dir / "bond_summary.csv", index=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BOND DISTRIBUTION SUMMARY")
    logger.info(f"{'='*60}")

    for endpoint in df["endpoint"].unique():
        subset = df[df["endpoint"] == endpoint]
        logger.info(f"\n{endpoint} (n={len(subset)}):")
        logger.info(f"  Covalent:  {subset['covalent_ratio'].mean():.3f} +/- {subset['covalent_ratio'].std():.3f}")
        logger.info(f"  Hydrogen:  {subset['hydrogen_ratio'].mean():.3f} +/- {subset['hydrogen_ratio'].std():.3f}")
        logger.info(f"  VdW:       {subset['vdw_ratio'].mean():.3f} +/- {subset['vdw_ratio'].std():.3f}")
        logger.info(f"  Avg bonds: {subset['total_bonds'].mean():.1f}")

    # Correct vs incorrect
    if "is_correct" in df.columns and df["is_correct"].notna().any():
        correct = df[df["is_correct"] == True]
        incorrect = df[df["is_correct"] == False]
        logger.info(f"\nCorrect predictions (n={len(correct)}):")
        logger.info(f"  Covalent: {correct['covalent_ratio'].mean():.3f}")
        logger.info(f"  Hydrogen: {correct['hydrogen_ratio'].mean():.3f}")
        logger.info(f"  VdW:      {correct['vdw_ratio'].mean():.3f}")
        logger.info(f"\nIncorrect predictions (n={len(incorrect)}):")
        logger.info(f"  Covalent: {incorrect['covalent_ratio'].mean():.3f}")
        logger.info(f"  Hydrogen: {incorrect['hydrogen_ratio'].mean():.3f}")
        logger.info(f"  VdW:      {incorrect['vdw_ratio'].mean():.3f}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--output-dir", default=str(BONDS_DIR))
    parser.add_argument("--disable-signal", nargs="+", default=[],
                        help="Signals to disable: similarity, distance, markers, energy")
    args = parser.parse_args()

    analyze_all_traces(
        traces_dir=Path(args.traces_dir),
        output_dir=Path(args.output_dir),
        disabled_signals=args.disable_signal if args.disable_signal else None,
    )
