"""
Reasoning Prompts

Structured prompts that force substantive chain-of-thought reasoning
for molecular property prediction. Each prompt guides Claude through:
  1. Structural analysis
  2. Property assessment (using tools)
  3. Mechanistic reasoning (SAR, structural alerts)
  4. Prediction with confidence
  5. Verification / cross-checking
"""

# System prompt for all molecular reasoning tasks
SYSTEM_PROMPT = """You are an expert medicinal chemist evaluating molecules for ADMET properties.
You have access to computational chemistry tools. For each molecule, you must:

1. ANALYZE the molecular structure — identify key functional groups, ring systems, pharmacophores, and structural alerts
2. COMPUTE properties using the available tools to gather quantitative data
3. REASON about the structure-activity relationship — explain WHY specific structural features would affect the property
4. PREDICT the outcome with a confidence level (high/medium/low)
5. VERIFY your reasoning — check for contradictions, consider alternative explanations, and note any uncertainty

Think step by step. Show your reasoning explicitly. Do not skip steps.
Your final answer must include:
- PREDICTION: positive or negative
- CONFIDENCE: high, medium, or low
- KEY_FACTORS: the top 3 structural features driving your prediction"""


# Per-endpoint prompts with context about the specific ADMET property
ENDPOINT_PROMPTS = {
    "cyp2d6_veith": """Evaluate whether this molecule is an inhibitor of CYP2D6 (cytochrome P450 2D6).

CYP2D6 metabolizes ~25% of marketed drugs. Key factors for CYP2D6 inhibition:
- Basic nitrogen atoms (protonatable at physiological pH)
- Lipophilic aromatic rings
- Hydrogen bond donors near basic center
- Molecular weight typically 200-500 Da
- Structural alerts: tertiary amines, quinidine-like scaffolds

Molecule SMILES: {smiles}

Analyze this molecule systematically using the available tools, then predict whether it inhibits CYP2D6.""",

    "cyp3a4_veith": """Evaluate whether this molecule is an inhibitor of CYP3A4 (cytochrome P450 3A4).

CYP3A4 has the broadest substrate specificity of CYP enzymes, metabolizing ~50% of drugs. Key factors:
- Large, flexible molecules tend to be substrates/inhibitors
- Lipophilicity (high LogP)
- Nitrogen-containing heterocycles (azoles, especially)
- Multiple aromatic rings
- Structural alerts: ketoconazole-like scaffolds, macrolides

Molecule SMILES: {smiles}

Analyze this molecule systematically using the available tools, then predict whether it inhibits CYP3A4.""",

    "cyp2c9_veith": """Evaluate whether this molecule is an inhibitor of CYP2C9 (cytochrome P450 2C9).

CYP2C9 metabolizes ~15% of drugs including warfarin and NSAIDs. Key factors:
- Acidic or neutral molecules preferred
- Aromatic rings with electronegative substituents
- Moderate lipophilicity
- Hydrogen bond acceptors
- Structural alerts: sulfonamides, coumarins, oxicams

Molecule SMILES: {smiles}

Analyze this molecule systematically using the available tools, then predict whether it inhibits CYP2C9.""",

    "herg": """Evaluate whether this molecule is a blocker of the hERG potassium channel.

hERG blockade causes QT prolongation and cardiac arrhythmia risk. Key factors:
- Basic nitrogen (cationic at physiological pH)
- Hydrophobic/aromatic domains flanking the basic center
- Molecular length/shape fitting the hERG channel cavity
- 3D conformation matters significantly (not just 2D structure)
- Structural alerts: terfenadine-like, piperidine + aromatic

Molecule SMILES: {smiles}

Analyze this molecule systematically using the available tools. Note: hERG prediction is particularly challenging as it depends on 3D molecular conformation. Predict whether it blocks the hERG channel.""",

    "pgp_broccatelli": """Evaluate whether this molecule is a substrate of P-glycoprotein (P-gp/MDR1).

P-gp is an efflux transporter affecting oral bioavailability and BBB penetration. Key factors:
- Amphipathic molecules (both hydrophilic and hydrophobic regions)
- Molecular weight > 400 Da increases likelihood
- Multiple hydrogen bond donors/acceptors
- Flexibility (rotatable bonds)
- Structural alerts: Verapamil-like, multiple aromatic rings

Molecule SMILES: {smiles}

Analyze this molecule systematically using the available tools, then predict whether it is a P-gp substrate.""",
}


# Minimal prompt for ablation study (no structural guidance)
MINIMAL_PROMPT = """Predict whether this molecule ({endpoint_name}) is positive or negative.

Molecule SMILES: {smiles}

Give your prediction (positive/negative) and confidence (high/medium/low)."""


def get_prompt(endpoint: str, smiles: str, minimal: bool = False) -> str:
    """Get the formatted prompt for a molecule and endpoint."""
    if minimal:
        endpoint_name = {
            "cyp2d6_veith": "CYP2D6 inhibitor",
            "cyp3a4_veith": "CYP3A4 inhibitor",
            "cyp2c9_veith": "CYP2C9 inhibitor",
            "herg": "hERG blocker",
            "pgp_broccatelli": "P-gp substrate",
        }[endpoint]
        return MINIMAL_PROMPT.format(endpoint_name=endpoint_name, smiles=smiles)
    return ENDPOINT_PROMPTS[endpoint].format(smiles=smiles)
