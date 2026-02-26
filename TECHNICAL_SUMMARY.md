# One-Page Summary: Molecular Targets Shape Reasoning Topology

**Ari Harrison, NovoQuant Nexus — February 2026**

---

## Claim

Chen et al. (2026) showed that chemistry-inspired bond types (covalent, hydrogen, van der Waals) describe AI reasoning structure and predict training quality (**Direction A**). We show the reverse: the molecular target being reasoned about determines which bond types dominate (**Direction B**). Together, this establishes a bidirectional link between chemistry and reasoning topology.

## Method

We collected 478 reasoning traces from Claude (Opus 4.6) predicting ADMET endpoints across five targets (CYP2D6, CYP3A4, CYP2C9, hERG, P-gp). Since Claude is closed-weight, we built a **Semantic Bond Classifier** — a four-signal ensemble (sentence embeddings, positional distance, discourse markers, energy proxy) that classifies pairwise reasoning-step interactions into the three bond types without attention weights. Energy proxy uses E = -log(sim) + log(d), aligning with the energy state E in Chen et al.'s Gibbs-Boltzmann formulation. Ablation confirms semantic similarity is the dominant signal (13.2 pp shift when removed). We pre-registered five hypotheses and applied Bonferroni correction across 12 tests.

## Key Result (H5)

CYP2D6 reasoning (2D substructure-driven) produces significantly more **covalent bonds** than hERG reasoning (3D conformation-dependent), which produces significantly more **van der Waals bonds**:

| Comparison | Cohen's *d* | *p*-value | Bonferroni |
|---|---|---|---|
| CYP2D6 covalent > hERG covalent | **0.94** | 1.2 x 10^-9 | Pass |
| hERG VdW > CYP2D6 VdW | **1.75** | 3.9 x 10^-24 | Pass |

**Sensitivity analysis** confirms H5 is not an artifact of pairwise distance bias: the separation survives local window restriction (k=3,5,7) and distance-bin reweighting. Even at k=3 (~6 bonds/trace), VdW separation has d=0.79 (p < 10^-7).

## Honest Nulls (H1--H4)

Reasoning structure does **not** predict accuracy in aggregate (*r* = -0.021, logistic AUROC = 0.531). Structure reflects *what* the model reasons about, not *how well*.

## Cross-Model Validation

H5 **replicates on GPT-5.2** (d=1.11 cov, d=0.87 VdW, both p < 0.001) with balanced trace lengths (10.6 vs 13.4 steps, ratio 0.79x). H5 **inverts on DeepSeek-R1** (d=-0.75 cov, d=-1.36 VdW) due to 2.1x trace length asymmetry. Pattern: H5 holds when trace lengths are balanced (Claude 0.93x, GPT-5.2 0.79x) and fails when they are not (R1 2.09x).

## Cross-Method Divergence

Our classifier diverges from attention-based distributions on OpenThoughts-114k (*r* = -0.83), due to O(*n*^2) pairwise bias on long traces (~242 steps). Sensitivity analysis shows this bias is not endpoint-correlated.

## Resources

- **Paper:** Full manuscript in `paper/` directory
- **Code + Data:** [github.com/realariharrison/molecular-reasoning-structure](https://github.com/realariharrison/molecular-reasoning-structure)
- **678 traces (478 Claude + 100 GPT-5.2 + 100 R1), bond classifications, sensitivity analysis, and all statistics** are publicly available
