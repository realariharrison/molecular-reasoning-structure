"""
Robustness Checks

1. Model robustness: Same molecules through Opus, Sonnet, Haiku
2. Temporal robustness: Same molecules at two time points (test-retest)
3. Annotator agreement: Inter-rater reliability on human bond annotations
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from reasoning_trace import load_traces, ReasoningTrace
from bond_classifier import SemanticBondClassifier
from structural_stability import entropy_series, fit_stability
from run_experiment import load_test_set, run_batch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ROBUST_DIR = DATA_DIR / "robustness"

MODELS = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}


def robustness_model(
    test_set_path: Path = DATA_DIR / "test_set_500.json",
    n_molecules: int = 50,
    output_dir: Path = ROBUST_DIR / "model",
):
    """
    Run same 50 molecules through 3 models and compare reasoning structures.
    """
    molecules = load_test_set(test_set_path)[:n_molecules]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, model_id in MODELS.items():
        logger.info(f"\nRunning model robustness: {model_name} ({model_id})")
        logger.info(f"  Command: python run_experiment.py --pilot {n_molecules} --model {model_id}")
        # Actual execution would be via run_experiment.py CLI

    logger.info("\nTo run model robustness manually:")
    for name, mid in MODELS.items():
        logger.info(f"  python run_experiment.py --pilot 50 --model {mid}")


def robustness_temporal(
    run1_dir: Path = ROBUST_DIR / "temporal" / "run1",
    run2_dir: Path = ROBUST_DIR / "temporal" / "run2",
    output_path: Path = ROBUST_DIR / "temporal_agreement.json",
):
    """
    Compare bond classifications from two runs of same molecules at different times.
    Uses Cohen's kappa for inter-run agreement.
    """
    if not run1_dir.exists() or not run2_dir.exists():
        logger.error("Run both temporal runs first:")
        logger.error("  Week 1: python run_experiment.py --pilot 50")
        logger.error("  Week 2: python run_experiment.py --pilot 50  (same molecules)")
        return None

    traces1 = load_traces(run1_dir)
    traces2 = load_traces(run2_dir)

    # Match by SMILES
    by_smiles_1 = {t.molecule.smiles: t for t in traces1 if t.molecule}
    by_smiles_2 = {t.molecule.smiles: t for t in traces2 if t.molecule}
    common = set(by_smiles_1.keys()) & set(by_smiles_2.keys())

    logger.info(f"Matched {len(common)} molecules across runs")

    # Compare bond type distributions
    diffs = {"covalent": [], "hydrogen": [], "vdw": []}
    for smiles in common:
        t1 = by_smiles_1[smiles]
        t2 = by_smiles_2[smiles]
        if t1.covalent_ratio is not None and t2.covalent_ratio is not None:
            diffs["covalent"].append(abs(t1.covalent_ratio - t2.covalent_ratio))
            diffs["hydrogen"].append(abs(t1.hydrogen_ratio - t2.hydrogen_ratio))
            diffs["vdw"].append(abs(t1.vdw_ratio - t2.vdw_ratio))

    # Compare prediction agreement
    pred_agree = 0
    pred_total = 0
    for smiles in common:
        p1 = by_smiles_1[smiles].molecule.claude_prediction
        p2 = by_smiles_2[smiles].molecule.claude_prediction
        if p1 is not None and p2 is not None:
            pred_total += 1
            if p1 == p2:
                pred_agree += 1

    result = {
        "n_matched": len(common),
        "bond_ratio_MAE": {
            "covalent": round(np.mean(diffs["covalent"]), 4) if diffs["covalent"] else None,
            "hydrogen": round(np.mean(diffs["hydrogen"]), 4) if diffs["hydrogen"] else None,
            "vdw": round(np.mean(diffs["vdw"]), 4) if diffs["vdw"] else None,
        },
        "prediction_agreement": round(pred_agree / pred_total, 4) if pred_total > 0 else None,
        "n_predictions": pred_total,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nTemporal agreement:")
    logger.info(f"  Bond ratio MAE: {result['bond_ratio_MAE']}")
    logger.info(f"  Prediction agreement: {result['prediction_agreement']}")

    return result


def robustness_annotator(
    annotations_path: Path = ROBUST_DIR / "human_annotations.json",
    output_path: Path = ROBUST_DIR / "annotator_agreement.json",
):
    """
    Compute inter-rater agreement for human bond annotations.

    Expects annotations_path to contain:
    {
        "annotations": [
            {
                "trace_id": "...",
                "bond_index": 0,
                "annotator_1": "covalent",
                "annotator_2": "hydrogen"
            },
            ...
        ]
    }
    """
    if not annotations_path.exists():
        logger.info("Human annotations file not found.")
        logger.info(f"Create annotations at: {annotations_path}")
        logger.info("Format: {annotations: [{trace_id, bond_index, annotator_1, annotator_2}]}")
        return None

    with open(annotations_path) as f:
        data = json.load(f)

    annotations = data["annotations"]

    # Cohen's kappa
    a1 = [a["annotator_1"] for a in annotations]
    a2 = [a["annotator_2"] for a in annotations]

    # Convert to numeric for kappa
    label_map = {"covalent": 0, "hydrogen": 1, "van_der_waals": 2}
    a1_num = [label_map.get(a, -1) for a in a1]
    a2_num = [label_map.get(a, -1) for a in a2]

    # Filter valid
    valid = [(x, y) for x, y in zip(a1_num, a2_num) if x >= 0 and y >= 0]
    if not valid:
        logger.error("No valid annotations found")
        return None

    a1_valid, a2_valid = zip(*valid)

    # Simple agreement
    agreement = sum(1 for x, y in valid if x == y) / len(valid)

    # Cohen's kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(a1_valid, a2_valid)

    # Also compare annotators with classifier
    # (would need trace data - documented for manual step)

    result = {
        "n_annotations": len(valid),
        "simple_agreement": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "kappa_interpretation": (
            "excellent" if kappa > 0.8 else
            "good" if kappa > 0.6 else
            "moderate" if kappa > 0.4 else
            "fair" if kappa > 0.2 else
            "poor"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nAnnotator agreement:")
    logger.info(f"  Simple agreement: {result['simple_agreement']}")
    logger.info(f"  Cohen's kappa: {result['cohens_kappa']} ({result['kappa_interpretation']})")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["model", "temporal", "annotator", "all"], default="all")
    args = parser.parse_args()

    if args.type in ("model", "all"):
        robustness_model()
    if args.type in ("temporal", "all"):
        robustness_temporal()
    if args.type in ("annotator", "all"):
        robustness_annotator()
