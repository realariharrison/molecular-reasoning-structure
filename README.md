# Molecular Targets Shape Reasoning Topology

**Bidirectional Validation of Chemistry-Inspired Bond Classification for AI Reasoning Chains**

Ari Harrison, NovoQuant Nexus (February 2026)

---

## Key Finding

Molecular targets determine AI reasoning topology. CYP2D6 reasoning (2D substructure-driven) produces significantly more covalent bonds than hERG reasoning (3D conformation-dependent), with Cohen's d = 0.94 (p < 10⁻⁹). hERG reasoning shows more van der Waals bonds than CYP2D6, with d = 1.75 (p < 10⁻²³). Both survive Bonferroni correction.

This completes a bidirectional loop with [Chen et al. (2026), "The Molecular Structure of Thought"](https://arxiv.org/abs/2601.06002): they showed chemistry metaphors explain reasoning structure (Direction A); we show chemistry problems shape reasoning structure (Direction B).

## Dataset

- **478 chain-of-thought reasoning traces** from Claude Opus 4.6
- **5 ADMET endpoints**: CYP2D6, CYP3A4, CYP2C9, hERG, P-gp
- **Live MCP tools**: get_molecule_profile, predict_admet, calculate_properties, check_compliance, get_3d_properties, get_molecule_info
- Test molecules drawn from [TDC ADMET Benchmark](https://tdcommons.ai/)

## Repository Structure

```
├── paper/                          # LaTeX manuscript + figures
│   ├── molecular_reasoning_structure.tex
│   ├── references.bib
│   └── fig*.pdf                    # Publication figures (7)
├── data/
│   ├── traces/                     # 478 reasoning trace JSON files
│   ├── bonds/bond_summary.csv      # Per-trace bond ratios
│   ├── hypothesis_results.json     # All H1-H5 test statistics
│   ├── stability_scores.csv        # Structural stability (lambda) per trace
│   ├── classifier_validation.json  # OpenThoughts cross-validation results
│   ├── ablations/                  # Classifier signal ablation results
│   ├── test_set_500.json           # Curated test set (478 molecules)
│   ├── figures/                    # PNG versions of all figures
│   └── RESULTS_SUMMARY.md          # Comprehensive results overview
├── reasoning_trace.py              # Core data structures
├── bond_classifier.py              # Semantic Bond Classifier (4-signal ensemble)
├── discourse_markers.json          # 73 discourse markers (covalent/hydrogen/vdw)
├── curate_test_set.py              # Stratified TDC test set curation
├── prompts.py                      # Per-endpoint reasoning prompts
├── experiment_runner.py            # Claude CLI trace collection
├── run_experiment.py               # Batch runner with checkpointing
├── analyze_bonds.py                # Bond classification pipeline
├── structural_stability.py         # Entropy convergence (lambda)
├── statistical_analysis.py         # H1-H5 hypothesis tests
├── generate_figures.py             # Publication figures
├── validate_classifier.py          # OpenThoughts-114k cross-validation
├── ablations.py                    # Classifier signal ablation
├── robustness_checks.py            # Model/temporal/annotator robustness
└── requirements.txt                # Python dependencies
```

## Reproducing the Analysis

```bash
pip install -r requirements.txt

# Bond classification (requires sentence-transformers)
python analyze_bonds.py

# Structural stability scores
python structural_stability.py

# Hypothesis tests (H1-H5)
python statistical_analysis.py

# Generate figures
python generate_figures.py

# Classifier ablation
python ablations.py --type classifier

# OpenThoughts cross-validation
python validate_classifier.py --n-samples 200
```

## Collecting New Traces

Requires Claude Code CLI with MCP tools configured:

```bash
# Curate test set from TDC
python curate_test_set.py

# Pilot run (20 molecules)
python run_experiment.py --pilot 20

# Full run (all molecules)
python run_experiment.py
```

## Results Summary

| Hypothesis | Result | p-value | Effect Size | Significant? |
|---|---|---|---|---|
| H1: Stability → accuracy | r = −0.021 | 0.653 | — | No |
| H2: Per-endpoint correlations | r ∈ [−0.17, 0.15] | >0.12 | — | No |
| H3: Bond ratios → accuracy | r < 0.05 | >0.26 | — | No |
| H4: Correct vs incorrect stability | d = −0.049 | 0.674 | — | No |
| **H5a: CYP2D6 cov > hERG cov** | **t = 6.29** | **<10⁻⁹** | **d = 0.94** | **Yes** |
| **H5b: hERG vdw > CYP2D6 vdw** | **t = 11.70** | **<10⁻²³** | **d = 1.75** | **Yes** |

## Citation

```bibtex
@article{harrison2026molecular,
  title={Molecular Targets Shape Reasoning Topology: Bidirectional Validation
         of Chemistry-Inspired Bond Classification for AI Reasoning Chains},
  author={Harrison, Ari},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See individual data files for source attribution (TDC benchmark data).
