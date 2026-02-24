# One-Page Summary: Molecular Targets Shape Reasoning Topology

**Ari Harrison, NovoQuant Nexus — February 2026**

---

## Claim

Chen et al. (2026) showed that chemistry-inspired bond types (covalent, hydrogen, van der Waals) describe AI reasoning structure and predict training quality (**Direction A**). We show the reverse: the molecular target being reasoned about determines which bond types dominate (**Direction B**). Together, this establishes a bidirectional link between chemistry and reasoning topology.

## Method

We collected 478 reasoning traces from Claude (Opus 4.6) predicting ADMET endpoints across five targets (CYP2D6, CYP3A4, CYP2C9, hERG, P-gp). Since Claude is closed-weight, we built a **Semantic Bond Classifier** — a four-signal ensemble (sentence embeddings, positional distance, discourse markers, energy proxy) that classifies pairwise reasoning-step interactions into your three bond types without attention weights. Ablation confirms semantic similarity is the dominant signal (13.2 pp shift when removed). We pre-registered five hypotheses and applied Bonferroni correction across 12 tests.

## Key Result (H5)

CYP2D6 reasoning (2D substructure-driven) produces significantly more **covalent bonds** than hERG reasoning (3D conformation-dependent), which produces significantly more **van der Waals bonds**:

| Comparison | Cohen's *d* | *p*-value | Bonferroni |
|---|---|---|---|
| CYP2D6 covalent > hERG covalent | **0.94** | 1.2 x 10^-9 | Pass |
| hERG VdW > CYP2D6 VdW | **1.75** | 3.9 x 10^-24 | Pass |

## Honest Nulls (H1--H4)

Reasoning structure does **not** predict accuracy in aggregate (*r* = -0.021, logistic AUROC = 0.531). Structure reflects *what* the model reasons about, not *how well*.

## Cross-Method Divergence

Our classifier diverges from your attention-based distributions on OpenThoughts-114k (*r* = -0.764), due to O(*n*^2) pairwise bias on long traces (~242 steps). Our primary analysis uses shorter Claude traces (~22 steps) where this bias is minimal, and H5 is a *relative* between-endpoint difference robust to baseline shift. We discuss this openly in Section 4.6.

## Resources

- **Paper:** Full manuscript in `paper/` directory
- **Code + Data:** [github.com/realariharrison/molecular-reasoning-structure](https://github.com/realariharrison/molecular-reasoning-structure)
- **478 traces, bond classifications, all statistics, and figures** are publicly available
