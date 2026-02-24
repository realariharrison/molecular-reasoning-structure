"""
Ablation Studies

Tests the contribution of each component to the overall result.

1. Prompt ablation: minimal vs full reasoning prompt
2. Tool ablation: restricted vs full tool set
3. Classifier ablation: remove each of 4 classification signals
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from reasoning_trace import load_traces, ReasoningTrace
from bond_classifier import SemanticBondClassifier
from structural_stability import entropy_series, fit_stability
from run_experiment import load_test_set, run_batch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ABLATION_DIR = DATA_DIR / "ablations"


def ablation_classifier_signals(
    traces_dir: Path = DATA_DIR / "traces",
    output_path: Path = ABLATION_DIR / "classifier_signals.json",
):
    """
    Ablation: Remove each classifier signal and measure impact on bond distributions.

    Signals: similarity, distance, markers, energy
    """
    traces = load_traces(traces_dir)
    logger.info(f"Running classifier ablation on {len(traces)} traces")

    results = {}

    # Full model (baseline)
    classifier = SemanticBondClassifier()
    full_classified = classifier.classify_traces(traces)
    results["full"] = _summarize_traces(full_classified)

    # Remove each signal
    for signal in ["similarity", "distance", "markers", "energy"]:
        logger.info(f"  Ablating: {signal}")
        classifier = SemanticBondClassifier(disabled_signals=[signal])
        # Reload traces fresh (embeddings get modified)
        traces = load_traces(traces_dir)
        ablated = classifier.classify_traces(traces)
        results[f"no_{signal}"] = _summarize_traces(ablated)

    # Compute deltas
    baseline = results["full"]
    for key in results:
        if key == "full":
            continue
        for metric in ["covalent_ratio", "hydrogen_ratio", "vdw_ratio"]:
            delta = results[key][metric] - baseline[metric]
            results[key][f"{metric}_delta"] = round(delta, 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nClassifier ablation saved to {output_path}")
    return results


def ablation_prompt(
    test_set_path: Path = DATA_DIR / "test_set_500.json",
    n_molecules: int = 100,
    output_dir: Path = ABLATION_DIR / "prompt",
):
    """
    Ablation: Run 100 molecules with minimal prompt vs full prompt.
    Full prompt results should already exist; this runs minimal only.
    """
    molecules = load_test_set(test_set_path)[:n_molecules]

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"

    logger.info(f"Running prompt ablation: {n_molecules} molecules (minimal prompt)")
    run_batch(
        molecules=molecules,
        checkpoint_every=25,
        minimal_prompt=True,
    )

    # The traces are saved to DATA_DIR/traces by default
    # For ablation, we'd need to redirect output - documented for manual execution


def ablation_tools(
    test_set_path: Path = DATA_DIR / "test_set_500.json",
    n_molecules: int = 100,
    output_dir: Path = ABLATION_DIR / "tools",
):
    """
    Ablation: Run with restricted tool sets.

    Configurations:
    - full: all 5 tools (baseline)
    - no_3d: remove get_3d_properties
    - no_compliance: remove check_compliance
    - minimal: only calculate_properties + predict_admet
    """
    molecules = load_test_set(test_set_path)[:n_molecules]

    tool_configs = {
        "no_3d": ["get_molecule_profile", "predict_admet", "calculate_properties", "check_compliance"],
        "no_compliance": ["get_molecule_profile", "predict_admet", "calculate_properties", "get_3d_properties"],
        "minimal": ["calculate_properties", "predict_admet"],
    }

    for config_name, tools in tool_configs.items():
        logger.info(f"\nTool ablation: {config_name} ({tools})")
        # These would need to be run separately with --tools flag
        # Documented for manual execution via run_experiment.py

    logger.info("\nTo run tool ablations manually:")
    for config_name, tools in tool_configs.items():
        tool_str = " ".join(tools)
        logger.info(f"  python run_experiment.py --pilot 100 --tools {tool_str}")


def _summarize_traces(traces: list[ReasoningTrace]) -> dict:
    """Compute summary statistics for classified traces."""
    cov = [t.covalent_ratio for t in traces if t.covalent_ratio is not None]
    hyd = [t.hydrogen_ratio for t in traces if t.hydrogen_ratio is not None]
    vdw = [t.vdw_ratio for t in traces if t.vdw_ratio is not None]
    bonds = [t.total_bonds for t in traces]

    # Stability
    stabilities = []
    for t in traces:
        ent = entropy_series(t)
        s = fit_stability(ent)
        if s is not None:
            stabilities.append(s)

    return {
        "n_traces": len(traces),
        "covalent_ratio": round(np.mean(cov), 4) if cov else 0.0,
        "hydrogen_ratio": round(np.mean(hyd), 4) if hyd else 0.0,
        "vdw_ratio": round(np.mean(vdw), 4) if vdw else 0.0,
        "avg_bonds": round(np.mean(bonds), 1) if bonds else 0.0,
        "avg_stability": round(np.mean(stabilities), 4) if stabilities else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["classifier", "prompt", "tools", "all"], default="classifier")
    args = parser.parse_args()

    if args.type in ("classifier", "all"):
        ablation_classifier_signals()
    if args.type in ("prompt", "all"):
        ablation_prompt()
    if args.type in ("tools", "all"):
        ablation_tools()
