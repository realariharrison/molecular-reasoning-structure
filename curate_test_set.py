from __future__ import annotations
"""
Test Set Curation

Curates a stratified, diversity-filtered set of 500 molecules from TDC benchmarks.
100 per endpoint (50 positive + 50 negative), Tanimoto distance > 0.3 between pairs.

Endpoints: CYP2D6, CYP3A4, CYP2C9, hERG, P-gp

Reuses TDC loading pattern from novoexpert1-tdc-benchmark/run_benchmark.py.
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

logger = logging.getLogger(__name__)

# TDC endpoint names matching run_tdc_benchmark.py
ENDPOINTS = {
    "cyp2d6_veith": {"name": "CYP2D6", "n_per_class": 50},
    "cyp3a4_veith": {"name": "CYP3A4", "n_per_class": 50},
    "cyp2c9_veith": {"name": "CYP2C9", "n_per_class": 50},
    "herg":         {"name": "hERG",    "n_per_class": 50},
    "pgp_broccatelli": {"name": "P-gp", "n_per_class": 50},
}


def compute_tanimoto_matrix(smiles_list: list[str]) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix using Morgan fingerprints."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    n = len(fps)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i + 1, n):
            if fps[j] is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def diversity_filter(
    df: pd.DataFrame, n_target: int, max_similarity: float = 0.7, seed: int = 42
) -> pd.DataFrame:
    """
    Greedy diversity selection: pick molecules with pairwise Tanimoto < max_similarity.
    max_similarity = 0.7 means Tanimoto distance > 0.3.
    """
    rng = np.random.RandomState(seed)

    # Shuffle first to avoid ordering bias
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(df) <= n_target:
        return df

    smiles_list = df["smiles"].tolist()
    sim_matrix = compute_tanimoto_matrix(smiles_list)

    selected = [0]  # Start with first molecule
    for candidate in range(1, len(df)):
        if len(selected) >= n_target:
            break
        # Check similarity against all selected
        max_sim = max(sim_matrix[candidate, s] for s in selected)
        if max_sim < max_similarity:
            selected.append(candidate)

    # If not enough, relax and add remaining (sorted by max diversity)
    if len(selected) < n_target:
        remaining = [i for i in range(len(df)) if i not in set(selected)]
        # Sort by minimum max-similarity to selected set
        remaining_scores = []
        for idx in remaining:
            max_sim = max(sim_matrix[idx, s] for s in selected)
            remaining_scores.append((idx, max_sim))
        remaining_scores.sort(key=lambda x: x[1])
        for idx, _ in remaining_scores:
            if len(selected) >= n_target:
                break
            selected.append(idx)

    return df.iloc[selected].reset_index(drop=True)


def curate_endpoint(
    group, endpoint: str, config: dict, seed: int = 42
) -> pd.DataFrame:
    """Curate stratified, diversity-filtered molecules for one endpoint."""
    benchmark = group.get(endpoint)
    test = benchmark["test"]

    # TDC uses 'Drug' for SMILES and 'Y' for labels
    df = pd.DataFrame({"smiles": test["Drug"], "label": test["Y"]})

    # Validate it's binary classification
    unique_labels = df["label"].unique()
    logger.info(f"  {endpoint}: {len(df)} test molecules, labels: {sorted(unique_labels)}")

    n_per_class = config["n_per_class"]

    # Stratified sampling
    positives = df[df["label"] == 1].copy()
    negatives = df[df["label"] == 0].copy()

    logger.info(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Diversity filter within each class
    pos_selected = diversity_filter(positives, n_per_class, seed=seed)
    neg_selected = diversity_filter(negatives, n_per_class, seed=seed)

    logger.info(f"  Selected: {len(pos_selected)} pos, {len(neg_selected)} neg")

    combined = pd.concat([pos_selected, neg_selected], ignore_index=True)
    combined["endpoint"] = endpoint
    combined["endpoint_name"] = config["name"]

    return combined


def curate_test_set(
    data_dir: str = "data/tdc_cache",
    output_path: str = "data/test_set_500.json",
    seed: int = 42,
):
    """Curate full 500-molecule test set across all endpoints."""
    data_dir = Path(__file__).parent / data_dir
    output_path = Path(__file__).parent / output_path
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TDC ADMET benchmark group...")
    group = admet_group(path=str(data_dir))

    all_molecules = []
    for endpoint, config in ENDPOINTS.items():
        logger.info(f"\nCurating {endpoint} ({config['name']})...")
        df = curate_endpoint(group, endpoint, config, seed=seed)
        all_molecules.append(df)

    combined = pd.concat(all_molecules, ignore_index=True)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Total molecules: {len(combined)}")
    for endpoint in ENDPOINTS:
        subset = combined[combined["endpoint"] == endpoint]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        logger.info(f"  {endpoint}: {len(subset)} ({n_pos} pos, {n_neg} neg)")

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = combined.to_dict(orient="records")
    output = {
        "n_molecules": len(combined),
        "n_endpoints": len(ENDPOINTS),
        "endpoints": list(ENDPOINTS.keys()),
        "seed": seed,
        "molecules": records,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSaved to {output_path}")
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Curate TDC test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/test_set_500.json")
    args = parser.parse_args()

    curate_test_set(output_path=args.output, seed=args.seed)
