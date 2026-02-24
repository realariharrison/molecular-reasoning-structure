"""
Reasoning Trace Data Structures

Captures Claude's chain-of-thought between tool calls for bond classification.
Mirrors the decision logging pattern from quanta-mcp/ai/decision_logger.py.
"""

from __future__ import annotations

import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class StepType(str, Enum):
    """Type of reasoning step in a trace."""
    REASONING = "reasoning"       # Free-form reasoning text
    TOOL_CALL = "tool_call"       # Tool invocation
    TOOL_RESULT = "tool_result"   # Tool response
    CONCLUSION = "conclusion"     # Final prediction


class BondType(str, Enum):
    """Molecular bond types for reasoning structure classification."""
    COVALENT = "covalent"           # Deep deductive reasoning
    HYDROGEN = "hydrogen"           # Self-reflection / verification
    VAN_DER_WAALS = "van_der_waals" # Exploratory / tangential


@dataclass
class ReasoningStep:
    """A single step in a reasoning trace."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_index: int = 0
    step_type: StepType = StepType.REASONING
    content: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Populated by bond classifier
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        # Don't serialize embeddings to JSON traces (large)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ReasoningStep:
        d = d.copy()
        d["step_type"] = StepType(d["step_type"])
        d.pop("embedding", None)
        return cls(**d)


@dataclass
class Bond:
    """A classified bond between two reasoning steps."""
    source_index: int
    target_index: int
    bond_type: BondType
    similarity: float        # Cosine similarity between embeddings
    distance: int            # Positional distance |target - source|
    energy: float            # Energy proxy: -log(sim) * (1/dist)
    marker_type: Optional[str] = None  # Discourse marker that triggered classification
    confidence: float = 0.0  # Ensemble confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bond_type"] = self.bond_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Bond:
        d = d.copy()
        d["bond_type"] = BondType(d["bond_type"])
        return cls(**d)


@dataclass
class MoleculeResult:
    """Ground truth and predictions for a single molecule."""
    smiles: str
    endpoint: str               # e.g., "cyp2d6_veith"
    ground_truth: int           # TDC binary label (0 or 1)
    claude_prediction: Optional[int] = None      # Claude's binary prediction
    claude_confidence: Optional[str] = None       # high/medium/low
    novoexpert1_prediction: Optional[float] = None  # Chemprop probability

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MoleculeResult:
        return cls(**d)


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for one molecule evaluation."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    molecule: Optional[MoleculeResult] = None
    steps: list[ReasoningStep] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    model: str = "claude-opus-4-6"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Computed metrics (populated by analysis)
    structural_stability: Optional[float] = None
    covalent_ratio: Optional[float] = None
    hydrogen_ratio: Optional[float] = None
    vdw_ratio: Optional[float] = None
    total_bonds: int = 0

    @property
    def reasoning_steps(self) -> list[ReasoningStep]:
        """Only text reasoning steps (not tool calls/results)."""
        return [s for s in self.steps if s.step_type == StepType.REASONING]

    @property
    def num_reasoning_steps(self) -> int:
        return len(self.reasoning_steps)

    @property
    def num_tool_calls(self) -> int:
        return len([s for s in self.steps if s.step_type == StepType.TOOL_CALL])

    @property
    def is_correct(self) -> Optional[bool]:
        if self.molecule is None or self.molecule.claude_prediction is None:
            return None
        return self.molecule.claude_prediction == self.molecule.ground_truth

    def add_step(self, step_type: StepType, content: str,
                 tool_name: str | None = None,
                 tool_input: dict | None = None,
                 tool_output: str | None = None) -> ReasoningStep:
        step = ReasoningStep(
            step_index=len(self.steps),
            step_type=step_type,
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        )
        self.steps.append(step)
        return step

    def compute_bond_ratios(self):
        """Compute bond type ratios from classified bonds."""
        self.total_bonds = len(self.bonds)
        if self.total_bonds == 0:
            self.covalent_ratio = 0.0
            self.hydrogen_ratio = 0.0
            self.vdw_ratio = 0.0
            return
        cov = sum(1 for b in self.bonds if b.bond_type == BondType.COVALENT)
        hyd = sum(1 for b in self.bonds if b.bond_type == BondType.HYDROGEN)
        vdw = sum(1 for b in self.bonds if b.bond_type == BondType.VAN_DER_WAALS)
        self.covalent_ratio = cov / self.total_bonds
        self.hydrogen_ratio = hyd / self.total_bonds
        self.vdw_ratio = vdw / self.total_bonds

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "molecule": self.molecule.to_dict() if self.molecule else None,
            "steps": [s.to_dict() for s in self.steps],
            "bonds": [b.to_dict() for b in self.bonds],
            "model": self.model,
            "created_at": self.created_at,
            "structural_stability": self.structural_stability,
            "covalent_ratio": self.covalent_ratio,
            "hydrogen_ratio": self.hydrogen_ratio,
            "vdw_ratio": self.vdw_ratio,
            "total_bonds": self.total_bonds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReasoningTrace:
        trace = cls(
            trace_id=d["trace_id"],
            model=d.get("model", "claude-opus-4-6"),
            created_at=d.get("created_at", ""),
            structural_stability=d.get("structural_stability"),
            covalent_ratio=d.get("covalent_ratio"),
            hydrogen_ratio=d.get("hydrogen_ratio"),
            vdw_ratio=d.get("vdw_ratio"),
            total_bonds=d.get("total_bonds", 0),
        )
        if d.get("molecule"):
            trace.molecule = MoleculeResult.from_dict(d["molecule"])
        trace.steps = [ReasoningStep.from_dict(s) for s in d.get("steps", [])]
        trace.bonds = [Bond.from_dict(b) for b in d.get("bonds", [])]
        return trace

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ReasoningTrace:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def load_traces(directory: Path) -> list[ReasoningTrace]:
    """Load all traces from a directory of JSON files."""
    traces = []
    for path in sorted(directory.glob("*.json")):
        traces.append(ReasoningTrace.load(path))
    return traces


def save_traces(traces: list[ReasoningTrace], directory: Path):
    """Save all traces to a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        filename = f"{trace.molecule.endpoint}_{trace.trace_id[:8]}.json" if trace.molecule else f"{trace.trace_id[:8]}.json"
        trace.save(directory / filename)
