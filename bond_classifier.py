"""
Semantic Bond Classifier

Four-signal ensemble for classifying reasoning bonds without attention weights.
Designed as a closed-source-model alternative to ByteDance's attention-based method.

Signals:
  1. Semantic similarity (sentence-transformers cosine similarity)
  2. Discourse markers (keyword matching)
  3. Positional distance (step index gap)
  4. Energy proxy (Gibbs-Boltzmann inspired: E = -log(sim) / dist)

References:
  - Chen et al. "The Molecular Structure of Thought" arXiv:2601.06002
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from reasoning_trace import (
    ReasoningTrace, ReasoningStep, Bond, BondType, StepType
)

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden for ablation)
DEFAULT_THRESHOLDS = {
    "covalent_max_dist": 2,
    "covalent_min_sim": 0.6,
    "hydrogen_min_dist": 3,
    "hydrogen_min_sim": 0.4,
    "vdw_min_dist": 3,
    "vdw_max_sim": 0.4,
    "min_step_length": 20,  # Minimum chars for a step to be considered
}


class SemanticBondClassifier:
    """
    Classifies bonds between reasoning steps using a 4-signal ensemble.

    This replaces ByteDance's attention-weight-based classification with
    a text-analysis approach that works with any model (including closed-source).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        markers_path: Optional[Path] = None,
        thresholds: Optional[dict] = None,
        disabled_signals: Optional[list[str]] = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.disabled_signals = set(disabled_signals or [])

        # Load discourse markers
        if markers_path is None:
            markers_path = Path(__file__).parent / "discourse_markers.json"
        with open(markers_path) as f:
            markers_data = json.load(f)

        self.markers = {
            BondType.COVALENT: [m.lower() for m in markers_data["covalent"]["markers"]],
            BondType.HYDROGEN: [m.lower() for m in markers_data["hydrogen"]["markers"]],
            BondType.VAN_DER_WAALS: [m.lower() for m in markers_data["van_der_waals"]["markers"]],
        }

    def classify_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Classify all bonds in a reasoning trace."""
        # Get reasoning steps only (filter out tool calls/results)
        reasoning_steps = [
            s for s in trace.steps
            if s.step_type in (StepType.REASONING, StepType.CONCLUSION)
            and len(s.content.strip()) >= self.thresholds["min_step_length"]
        ]

        if len(reasoning_steps) < 2:
            logger.warning(f"Trace {trace.trace_id}: <2 reasoning steps, skipping")
            trace.bonds = []
            trace.compute_bond_ratios()
            return trace

        # Compute embeddings
        texts = [s.content for s in reasoning_steps]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        for step, emb in zip(reasoning_steps, embeddings):
            step.embedding = emb.tolist()

        # Classify all pairwise bonds (i < j only)
        bonds = []
        for i in range(len(reasoning_steps)):
            for j in range(i + 1, len(reasoning_steps)):
                bond = self._classify_pair(
                    reasoning_steps[i], reasoning_steps[j],
                    embeddings[i], embeddings[j]
                )
                bonds.append(bond)

        trace.bonds = bonds
        trace.compute_bond_ratios()
        return trace

    def _classify_pair(
        self,
        step_i: ReasoningStep,
        step_j: ReasoningStep,
        emb_i: np.ndarray,
        emb_j: np.ndarray,
    ) -> Bond:
        """Classify the bond between two reasoning steps."""
        # Signal 1: Semantic similarity
        similarity = float(np.dot(emb_i, emb_j))  # Already normalized

        # Signal 2: Positional distance
        distance = abs(step_j.step_index - step_i.step_index)

        # Signal 3: Energy proxy
        sim_clamped = max(similarity, 1e-6)
        dist_clamped = max(distance, 1)
        energy = -math.log(sim_clamped) * (1.0 / dist_clamped)

        # Signal 4: Discourse markers in target step
        marker_scores = self._score_markers(step_j.content)

        # Ensemble classification
        bond_type, confidence, marker_type = self._ensemble_classify(
            similarity, distance, energy, marker_scores
        )

        return Bond(
            source_index=step_i.step_index,
            target_index=step_j.step_index,
            bond_type=bond_type,
            similarity=round(similarity, 4),
            distance=distance,
            energy=round(energy, 4),
            marker_type=marker_type,
            confidence=round(confidence, 4),
        )

    def _score_markers(self, text: str) -> dict[BondType, float]:
        """Count discourse marker matches per bond type."""
        text_lower = text.lower()
        scores = {}
        for bond_type, markers in self.markers.items():
            count = sum(1 for m in markers if m in text_lower)
            scores[bond_type] = count
        return scores

    def _ensemble_classify(
        self,
        similarity: float,
        distance: int,
        energy: float,
        marker_scores: dict[BondType, float],
    ) -> tuple[BondType, float, Optional[str]]:
        """
        Combine 4 signals into a bond classification.

        Returns (bond_type, confidence, dominant_marker_type).
        """
        t = self.thresholds
        votes = {BondType.COVALENT: 0.0, BondType.HYDROGEN: 0.0, BondType.VAN_DER_WAALS: 0.0}

        # Signal 1: Semantic similarity
        if "similarity" not in self.disabled_signals:
            if similarity > t["covalent_min_sim"]:
                votes[BondType.COVALENT] += similarity
            elif similarity > t["hydrogen_min_sim"]:
                votes[BondType.HYDROGEN] += similarity * 0.8
            else:
                votes[BondType.VAN_DER_WAALS] += (1 - similarity) * 0.5

        # Signal 2: Positional distance
        if "distance" not in self.disabled_signals:
            if distance <= t["covalent_max_dist"]:
                votes[BondType.COVALENT] += 0.4
            elif distance >= t["vdw_min_dist"]:
                if similarity < t["vdw_max_sim"]:
                    votes[BondType.VAN_DER_WAALS] += 0.4
                else:
                    votes[BondType.HYDROGEN] += 0.4

        # Signal 3: Discourse markers
        dominant_marker = None
        if "markers" not in self.disabled_signals:
            total_markers = sum(marker_scores.values())
            if total_markers > 0:
                for bt, count in marker_scores.items():
                    votes[bt] += (count / total_markers) * 0.6
                dominant_bt = max(marker_scores, key=marker_scores.get)
                if marker_scores[dominant_bt] > 0:
                    dominant_marker = dominant_bt.value

        # Signal 4: Energy proxy
        if "energy" not in self.disabled_signals:
            # Low energy = strong bond (covalent), high energy = weak bond (vdw)
            if energy < 0.5:
                votes[BondType.COVALENT] += 0.3
            elif energy < 1.5:
                votes[BondType.HYDROGEN] += 0.3
            else:
                votes[BondType.VAN_DER_WAALS] += 0.3

        # Winner-take-all
        winner = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[winner] / total_votes if total_votes > 0 else 0.0

        return winner, confidence, dominant_marker

    def classify_traces(self, traces: list[ReasoningTrace]) -> list[ReasoningTrace]:
        """Classify bonds in multiple traces."""
        classified = []
        for i, trace in enumerate(traces):
            logger.info(f"Classifying trace {i+1}/{len(traces)}: {trace.trace_id[:8]}")
            classified.append(self.classify_trace(trace))
        return classified
