"""
Classifier Validation

Validates SemanticBondClassifier against ByteDance's attention-based method
by running on OpenThoughts-3 traces and comparing bond distributions.

Target: Pearson r > 0.8 with their reported distributions.
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from datasets import load_dataset

from reasoning_trace import ReasoningTrace, ReasoningStep, StepType
from bond_classifier import SemanticBondClassifier

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def load_openthoughts_traces(n_samples: int = 200) -> list[ReasoningTrace]:
    """
    Load OpenThoughts-114k reasoning traces and convert to our trace format.

    The dataset uses conversations format with <|begin_of_thought|>...<|end_of_thought|>
    tags inside assistant turns.
    """
    logger.info(f"Loading OpenThoughts-114k (n={n_samples})...")

    dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)

    traces = []
    for i, example in enumerate(dataset):
        if len(traces) >= n_samples:
            break

        trace = ReasoningTrace(trace_id=f"ot3_{i:05d}")

        # Extract thought from conversations
        thought = ""
        for turn in example.get("conversations", []):
            if turn.get("from") == "assistant":
                val = turn.get("value", "")
                if "<|begin_of_thought|>" in val and "<|end_of_thought|>" in val:
                    start = val.index("<|begin_of_thought|>") + len("<|begin_of_thought|>")
                    end = val.index("<|end_of_thought|>")
                    thought = val[start:end].strip()
                    break

        if not thought:
            continue

        # Split into paragraphs as reasoning steps
        paragraphs = [p.strip() for p in thought.split("\n\n") if p.strip() and len(p.strip()) > 20]

        for j, para in enumerate(paragraphs):
            trace.add_step(
                step_type=StepType.REASONING,
                content=para,
            )

        if len(trace.steps) >= 2:
            traces.append(trace)

    logger.info(f"  Loaded {len(traces)} valid traces")
    return traces


# ByteDance reported distributions (from arXiv:2601.06002, Table 2)
# These are approximate from their paper figures
BYTEDANCE_REPORTED = {
    "openthoughts": {
        "covalent_ratio_mean": 0.45,
        "hydrogen_ratio_mean": 0.30,
        "vdw_ratio_mean": 0.25,
    },
}


def validate_against_bytedance(
    n_samples: int = 200,
    output_path: Path = DATA_DIR / "classifier_validation.json",
) -> dict:
    """
    Run our classifier on OpenThoughts traces and compare to ByteDance results.
    """
    # Load and classify
    traces = load_openthoughts_traces(n_samples)
    classifier = SemanticBondClassifier()
    classified = classifier.classify_traces(traces)

    # Compute our distributions
    ratios = {
        "covalent": [t.covalent_ratio for t in classified if t.covalent_ratio is not None],
        "hydrogen": [t.hydrogen_ratio for t in classified if t.hydrogen_ratio is not None],
        "vdw": [t.vdw_ratio for t in classified if t.vdw_ratio is not None],
    }

    our_means = {
        "covalent_ratio_mean": np.mean(ratios["covalent"]),
        "hydrogen_ratio_mean": np.mean(ratios["hydrogen"]),
        "vdw_ratio_mean": np.mean(ratios["vdw"]),
    }

    our_stds = {
        "covalent_ratio_std": np.std(ratios["covalent"]),
        "hydrogen_ratio_std": np.std(ratios["hydrogen"]),
        "vdw_ratio_std": np.std(ratios["vdw"]),
    }

    # Compare with ByteDance
    bd = BYTEDANCE_REPORTED["openthoughts"]
    our_vec = [our_means["covalent_ratio_mean"], our_means["hydrogen_ratio_mean"], our_means["vdw_ratio_mean"]]
    bd_vec = [bd["covalent_ratio_mean"], bd["hydrogen_ratio_mean"], bd["vdw_ratio_mean"]]

    # Guard against NaN/empty
    if any(np.isnan(v) for v in our_vec):
        logger.error(f"NaN in our distributions — only {len(classified)} traces classified")
        pearson_r, pearson_p = float("nan"), float("nan")
    else:
        pearson_r, pearson_p = stats.pearsonr(our_vec, bd_vec)

    result = {
        "n_traces": len(classified),
        "our_distributions": {k: float(v) for k, v in {**our_means, **our_stds}.items()},
        "bytedance_reported": bd,
        "pearson_r": float(round(pearson_r, 4)),
        "pearson_p": float(round(pearson_p, 6)),
        "target_r": 0.8,
        "meets_target": bool(pearson_r > 0.8),
        "per_trace_stats": {
            "n_valid": len(ratios["covalent"]),
            "avg_bonds_per_trace": float(np.mean([t.total_bonds for t in classified])),
            "avg_steps_per_trace": float(np.mean([t.num_reasoning_steps for t in classified])),
        },
    }

    logger.info(f"\nValidation Results:")
    logger.info(f"  Our means:  cov={our_means['covalent_ratio_mean']:.3f}, "
                f"h={our_means['hydrogen_ratio_mean']:.3f}, "
                f"vdw={our_means['vdw_ratio_mean']:.3f}")
    logger.info(f"  ByteDance:  cov={bd['covalent_ratio_mean']:.3f}, "
                f"h={bd['hydrogen_ratio_mean']:.3f}, "
                f"vdw={bd['vdw_ratio_mean']:.3f}")
    logger.info(f"  Pearson r:  {pearson_r:.4f} (target > 0.8)")
    logger.info(f"  Meets target: {result['meets_target']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    validate_against_bytedance(n_samples=args.n_samples)
