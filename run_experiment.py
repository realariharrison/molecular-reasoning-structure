#!/usr/bin/env python3
from __future__ import annotations
"""
Batch Experiment Runner

Runs 500 molecules through Claude with tool use, collecting reasoning traces.
Checkpoints every 50 molecules. Supports resume from checkpoint.

Usage:
    python run_experiment.py                          # Full run (500 molecules)
    python run_experiment.py --pilot 20               # Pilot run (20 molecules)
    python run_experiment.py --resume                  # Resume from checkpoint
    python run_experiment.py --endpoint cyp2d6_veith   # Single endpoint
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

from reasoning_trace import ReasoningTrace, load_traces, save_traces
from experiment_runner import run_molecule

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
TRACES_DIR = DATA_DIR / "traces"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"


def load_test_set(path: Path = DATA_DIR / "test_set_500.json") -> list[dict]:
    """Load curated test set."""
    with open(path) as f:
        data = json.load(f)
    return data["molecules"]


def load_checkpoint() -> dict:
    """Load checkpoint state."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_index": -1}


def save_checkpoint(state: dict):
    """Save checkpoint state."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2)


def run_batch(
    molecules: list[dict],
    model: str = "claude-opus-4-6",
    checkpoint_every: int = 50,
    delay: float = 2.0,
    resume: bool = False,
    restricted_tools: list[str] | None = None,
    minimal_prompt: bool = False,
):
    """
    Run batch experiment with checkpointing.

    Args:
        molecules: List of molecule dicts with smiles, endpoint, label
        model: Anthropic model ID
        checkpoint_every: Save checkpoint every N molecules
        delay: Seconds between API calls (rate limiting)
        resume: Resume from checkpoint
        restricted_tools: If set, only use these tool names (for ablation)
        minimal_prompt: Use minimal prompt (for ablation)
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    # Load or initialize checkpoint
    state = load_checkpoint() if resume else {"completed": [], "failed": [], "last_index": -1}
    completed_smiles = set(state["completed"])

    if restricted_tools:
        logger.info(f"Note: Tool restriction via CLI not supported; MCP tools are auto-available")

    total = len(molecules)
    start_time = time.time()
    traces_collected = 0

    logger.info(f"Starting batch run: {total} molecules, model={model}")
    logger.info(f"  Already completed: {len(completed_smiles)}")
    logger.info(f"  Checkpoint every: {checkpoint_every}")
    logger.info(f"  Output: {TRACES_DIR}")

    for i, mol in enumerate(molecules):
        smiles = mol["smiles"]
        endpoint = mol["endpoint"]
        label = int(mol["label"])

        # Skip already completed
        if smiles in completed_smiles:
            continue

        # Progress
        elapsed = time.time() - start_time
        rate = traces_collected / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        logger.info(
            f"[{i+1}/{total}] {endpoint} | "
            f"Completed: {traces_collected} | "
            f"Rate: {rate:.1f}/s | "
            f"ETA: {eta/60:.0f}min"
        )

        try:
            trace = run_molecule(
                smiles=smiles,
                endpoint=endpoint,
                ground_truth=label,
                model=model,
                minimal_prompt=minimal_prompt,
            )

            # Save individual trace
            filename = f"{endpoint}_{trace.trace_id[:8]}.json"
            trace.save(TRACES_DIR / filename)

            state["completed"].append(smiles)
            completed_smiles.add(smiles)
            traces_collected += 1

            # Log result
            pred = trace.molecule.claude_prediction
            correct = trace.is_correct
            n_steps = trace.num_reasoning_steps
            n_tools = trace.num_tool_calls
            logger.info(
                f"  -> pred={pred}, correct={correct}, "
                f"steps={n_steps}, tools={n_tools}"
            )

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            state["failed"].append({"smiles": smiles, "endpoint": endpoint, "error": str(e)})

        # Checkpoint
        state["last_index"] = i
        if (traces_collected % checkpoint_every == 0) and traces_collected > 0:
            save_checkpoint(state)
            logger.info(f"  Checkpoint saved ({traces_collected} traces)")

        # Rate limiting
        time.sleep(delay)

    # Final checkpoint
    save_checkpoint(state)

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH COMPLETE")
    logger.info(f"  Traces collected: {traces_collected}")
    logger.info(f"  Failed: {len(state['failed'])}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")
    logger.info(f"  Avg time per molecule: {elapsed/max(traces_collected,1):.1f}s")

    return state


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run reasoning trace experiment")
    parser.add_argument("--pilot", type=int, default=0, help="Pilot run with N molecules")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--endpoint", type=str, help="Run single endpoint only")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model ID")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between calls (seconds)")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--test-set", default="data/test_set_500.json")
    parser.add_argument("--minimal-prompt", action="store_true", help="Use minimal prompt (ablation)")
    parser.add_argument("--tools", nargs="+", help="Restricted tool names (ablation)")
    args = parser.parse_args()

    # Load test set
    test_set_path = Path(__file__).parent / args.test_set
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        logger.error("Run curate_test_set.py first.")
        sys.exit(1)

    molecules = load_test_set(test_set_path)
    logger.info(f"Loaded {len(molecules)} molecules from {test_set_path}")

    # Filter by endpoint
    if args.endpoint:
        molecules = [m for m in molecules if m["endpoint"] == args.endpoint]
        logger.info(f"Filtered to endpoint {args.endpoint}: {len(molecules)} molecules")

    # Pilot mode
    if args.pilot > 0:
        # Take balanced sample across endpoints
        from collections import defaultdict
        by_endpoint = defaultdict(list)
        for m in molecules:
            by_endpoint[m["endpoint"]].append(m)
        pilot_molecules = []
        per_endpoint = max(1, args.pilot // len(by_endpoint))
        for endpoint, mols in by_endpoint.items():
            pilot_molecules.extend(mols[:per_endpoint])
        molecules = pilot_molecules[:args.pilot]
        logger.info(f"Pilot mode: {len(molecules)} molecules")

    # Run
    state = run_batch(
        molecules=molecules,
        model=args.model,
        checkpoint_every=args.checkpoint_every,
        delay=args.delay,
        resume=args.resume,
        restricted_tools=args.tools,
        minimal_prompt=args.minimal_prompt,
    )

    # Print failure summary
    if state["failed"]:
        logger.info(f"\nFailed molecules ({len(state['failed'])}):")
        for f in state["failed"][:10]:
            logger.info(f"  {f['endpoint']}: {f['smiles'][:50]}... - {f['error']}")


if __name__ == "__main__":
    main()
