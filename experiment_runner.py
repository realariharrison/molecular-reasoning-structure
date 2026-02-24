"""
Experiment Runner

Executes reasoning trace collection using the `claude` CLI (Claude Code),
which authenticates via the user's subscription. Captures full chain-of-thought
between tool calls by parsing CLI output.

Tools are the live NovoMCP tools connected to Claude via MCP server.
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional

from reasoning_trace import (
    ReasoningTrace, ReasoningStep, MoleculeResult, StepType
)
from prompts import SYSTEM_PROMPT, get_prompt

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "claude-opus-4-6"


def extract_prediction(text: str) -> tuple[Optional[int], Optional[str]]:
    """Extract prediction and confidence from Claude's conclusion text."""
    text_lower = text.lower()
    # Strip markdown bold markers for cleaner matching
    text_clean = re.sub(r'\*+', '', text_lower)

    # Extract prediction — check structured format first, then natural language
    prediction = None

    # 1. Structured: "PREDICTION: positive/negative"
    if re.search(r"prediction:?\s*positive", text_clean):
        prediction = 1
    elif re.search(r"prediction:?\s*negative", text_clean):
        prediction = 0

    # 2. Explicit inhibitor/blocker/substrate language
    elif re.search(r"(is|likely|would be|predicted to be)\s+(an?\s+)?(strong\s+|moderate\s+|weak\s+)?(cyp\w*\s+)?inhibitor", text_clean):
        prediction = 1
    elif re.search(r"(not|unlikely|would not be|not predicted)\s+.{0,20}inhibitor", text_clean):
        prediction = 0
    elif re.search(r"(is|likely|would be|predicted to be)\s+(an?\s+)?(herg\s+)?(channel\s+)?blocker", text_clean):
        prediction = 1
    elif re.search(r"(not|unlikely|would not be|not predicted)\s+.{0,20}blocker", text_clean):
        prediction = 0
    elif re.search(r"(is|likely|would be|predicted to be)\s+(an?\s+)?(p-?gp?\s+)?substrate", text_clean):
        prediction = 1
    elif re.search(r"(not|unlikely|would not be|not predicted)\s+.{0,20}substrate", text_clean):
        prediction = 0

    # 3. Risk assessment language
    elif re.search(r"(high|significant|strong|substantial)\s+(risk|likelihood|probability|potential)\s+(of|for)\s+(inhibit|block|efflux|substrate)", text_clean):
        prediction = 1
    elif re.search(r"(low|minimal|negligible|unlikely)\s+(risk|likelihood|probability|potential)\s+(of|for)\s+(inhibit|block|efflux|substrate)", text_clean):
        prediction = 0
    elif re.search(r"strongly suggests?\s+(p-?gp\s+substrate|inhibit|block)", text_clean):
        prediction = 1
    elif re.search(r"does not\s+(suggest|indicate).{0,20}(inhibit|block|substrate)", text_clean):
        prediction = 0

    # 4. Verdict/conclusion patterns
    elif re.search(r"(verdict|conclusion|assessment|overall)[:\s]+.{0,30}(yes|positive|inhibitor|blocker|substrate)", text_clean):
        prediction = 1
    elif re.search(r"(verdict|conclusion|assessment|overall)[:\s]+.{0,30}(no|negative|non-?inhibitor|non-?blocker|non-?substrate)", text_clean):
        prediction = 0

    # 5. Last resort: count positive vs negative keywords in final text
    if prediction is None:
        pos_kw = len(re.findall(r"\b(inhibitor|blocker|substrate|positive|yes)\b", text_clean))
        neg_kw = len(re.findall(r"\b(non-?inhibitor|non-?blocker|non-?substrate|negative|no)\b", text_clean))
        if pos_kw > neg_kw and pos_kw >= 2:
            prediction = 1
        elif neg_kw > pos_kw and neg_kw >= 2:
            prediction = 0

    # Extract confidence
    confidence = None
    if re.search(r"confidence:?\s*high", text_clean):
        confidence = "high"
    elif re.search(r"confidence:?\s*medium|confidence:?\s*moderate", text_clean):
        confidence = "medium"
    elif re.search(r"confidence:?\s*low", text_clean):
        confidence = "low"
    # Natural language confidence
    elif re.search(r"(high|strong)\s+(confidence|certainty|conviction)", text_clean):
        confidence = "high"
    elif re.search(r"(moderate|medium|reasonable)\s+(confidence|certainty)", text_clean):
        confidence = "medium"
    elif re.search(r"(low|limited|uncertain)\s+(confidence|certainty)", text_clean):
        confidence = "low"

    return prediction, confidence


def parse_claude_output(raw_output: str) -> list[dict]:
    """
    Parse claude CLI output into reasoning steps.

    The CLI output contains interleaved reasoning text and tool call/result blocks.
    We split on structural markers to identify step boundaries.
    """
    steps = []

    # Split on double newlines to get paragraphs
    paragraphs = [p.strip() for p in raw_output.split("\n\n") if p.strip()]

    for para in paragraphs:
        # Detect tool calls (Claude Code formats them distinctively)
        if para.startswith("Tool:") or para.startswith("Using tool:") or "tool_use" in para.lower():
            steps.append({"type": "tool_call", "content": para})
        elif para.startswith("Result:") or para.startswith("Tool result:"):
            steps.append({"type": "tool_result", "content": para})
        else:
            # Check if conclusion
            if any(kw in para.lower() for kw in ["prediction:", "key_factors:", "final answer"]):
                steps.append({"type": "conclusion", "content": para})
            else:
                steps.append({"type": "reasoning", "content": para})

    return steps


def run_molecule(
    smiles: str,
    endpoint: str,
    ground_truth: int,
    model: str = DEFAULT_MODEL,
    minimal_prompt: bool = False,
    timeout: int = 300,
) -> ReasoningTrace:
    """
    Run a single molecule through Claude CLI, capturing the full reasoning trace.

    Uses `claude --print --model <model>` which authenticates via subscription.
    The MCP tools (NovoMCP) are available because they're configured in the user's
    Claude Code MCP settings.

    Args:
        smiles: SMILES string
        endpoint: TDC endpoint name
        ground_truth: Binary label (0 or 1)
        model: Model to use
        minimal_prompt: Use minimal prompt (for ablation)
        timeout: Max seconds per molecule

    Returns:
        ReasoningTrace with all steps captured
    """
    trace = ReasoningTrace(model=model)
    trace.molecule = MoleculeResult(
        smiles=smiles,
        endpoint=endpoint,
        ground_truth=ground_truth,
    )

    prompt = get_prompt(endpoint, smiles, minimal=minimal_prompt)

    # Build the full prompt with system instructions
    full_prompt = f"""{SYSTEM_PROMPT}

{prompt}

IMPORTANT: Use the available MCP tools (get_molecule_profile, predict_admet, calculate_properties, check_compliance, get_3d_properties) to gather data about this molecule. Then provide your structured analysis.

Your final answer MUST include these exact fields:
PREDICTION: positive or negative
CONFIDENCE: high, medium, or low
KEY_FACTORS: list your top 3 structural features"""

    # Run via claude CLI with --print (non-interactive, outputs result)
    # --allowedTools permits MCP tool calls without interactive approval
    # Prompt passed via stdin to avoid shell escaping issues
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--max-turns", "10",
        "--output-format", "json",
        "--allowedTools", "mcp__claude_ai_Novo__get_molecule_profile",
        "--allowedTools", "mcp__claude_ai_Novo__predict_admet",
        "--allowedTools", "mcp__claude_ai_Novo__calculate_properties",
        "--allowedTools", "mcp__claude_ai_Novo__check_compliance",
        "--allowedTools", "mcp__claude_ai_Novo__get_3d_properties",
        "--allowedTools", "mcp__claude_ai_Novo__get_molecule_info",
        "-p", full_prompt,
    ]

    # Must unset CLAUDECODE env var to allow nested invocation
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"Claude CLI error: {result.stderr[:500]}")
            raise RuntimeError(f"Claude CLI failed: {result.stderr[:200]}")

        raw_output = result.stdout

        # Try to parse JSON output format
        try:
            json_output = json.loads(raw_output)
            # JSON output format has a "result" field with the text
            if isinstance(json_output, dict):
                text_content = json_output.get("result", "")
                if not text_content:
                    text_content = json_output.get("content", "")
                if isinstance(text_content, list):
                    # Content blocks format
                    for block in text_content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                _add_text_steps(trace, block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                trace.add_step(
                                    step_type=StepType.TOOL_CALL,
                                    content=f"Calling {block.get('name', 'unknown')}",
                                    tool_name=block.get("name"),
                                    tool_input=block.get("input"),
                                )
                            elif block.get("type") == "tool_result":
                                content = block.get("content", "")
                                if isinstance(content, list):
                                    content = json.dumps(content)
                                trace.add_step(
                                    step_type=StepType.TOOL_RESULT,
                                    content=str(content)[:2000],
                                    tool_name=block.get("name"),
                                    tool_output=str(content)[:2000],
                                )
                elif isinstance(text_content, str):
                    _add_text_steps(trace, text_content)
                # Check for cost/usage info
                if "cost_usd" in json_output:
                    logger.info(f"  Cost: ${json_output['cost_usd']:.4f}")
            else:
                _add_text_steps(trace, raw_output)
        except json.JSONDecodeError:
            # Plain text output
            _add_text_steps(trace, raw_output)

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout after {timeout}s")
        raise RuntimeError(f"Timeout after {timeout}s")

    # Extract prediction — try conclusion steps first, then all reasoning
    conclusion_texts = [
        s.content for s in trace.steps
        if s.step_type == StepType.CONCLUSION
    ]
    reasoning_texts = [
        s.content for s in trace.steps
        if s.step_type == StepType.REASONING
    ]

    prediction, confidence = None, None

    # First try: conclusion steps only
    if conclusion_texts:
        combined = "\n".join(conclusion_texts)
        prediction, confidence = extract_prediction(combined)

    # Second try: last 5 reasoning steps
    if prediction is None and reasoning_texts:
        combined = "\n".join(reasoning_texts[-5:])
        prediction, confidence = extract_prediction(combined)

    # Third try: all reasoning text
    if prediction is None and reasoning_texts:
        combined = "\n".join(reasoning_texts)
        prediction, confidence = extract_prediction(combined)

    trace.molecule.claude_prediction = prediction
    trace.molecule.claude_confidence = confidence

    return trace


def _add_text_steps(trace: ReasoningTrace, text: str):
    """Parse text into reasoning steps and add to trace."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 10]
    for para in paragraphs:
        step_type = StepType.REASONING
        if any(kw in para.lower() for kw in ["prediction:", "key_factors:", "final answer"]):
            step_type = StepType.CONCLUSION
        trace.add_step(step_type=step_type, content=para)
