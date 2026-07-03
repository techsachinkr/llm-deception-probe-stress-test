# llm-deception-probe-stress-test

Full implementation of the experimental pipeline for the paper "Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations," published in the Proceedings of the Fifth Workshop on Generation, Evaluation and Metrics (GEM) at ACL 2026.

📄 **Paper:** https://aclanthology.org/2026.gem-main.43/ · **DOI:** [10.18653/v1/2026.gem-main.43](https://doi.org/10.18653/v1/2026.gem-main.43)

## Overview

This codebase implements four studies testing competing hypotheses about how deception is encoded in LLM activations:

| Study | Section | What it tests |
|-------|---------|---------------|
| **Study 1** | §5.1 | Scaling laws: emergence of linear separability across 1B→27B |
| **Study 2** | §5.2 | Geometry: is deception a direction, subspace, or cone? |
| **Study 3** | §5.3 | Adversarial robustness: five-probe battery against RL-obfuscation |
| **Study 4** | §5.4 | Entropy proxy: are probes detecting cognitive load? |


## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd deception_probes
pip install -r requirements.txt

huggingface-cli login

# 2. Full run on all models
python scripts/run_all.py

# 3. Or run specific studies
python scripts/run_all.py --models 1B 4B 12B 27B --studies 1 2
```

## Detailed Usage

### Command-Line Options

```bash
python scripts/run_all.py \
    --models 1B 4B 12B 27B \    # Which Gemma 3 sizes to test
    --studies 1 2 3 4 \          # Which studies to run
    --skip-datagen \             # Skip dataset generation (use cached)
    --skip-activation \          # Skip activation collection (use cached)
    --output-dir ./outputs \     # Where to save everything
    --seed 42                    # Random seed
```

In case of issues with 27B model:
```
pip install torchvision --force-reinstall --break-system-packages
```

### Additional experiments

#### 1A: Style-Augmented Probe Training

**Question:** Is distractor fragility fundamental or a training distribution artifact?

**Design:**
- Train styles: pirate, formal_academic, robotic, poetic
- Held-out test styles: shakespearean, childlike, slang_casual, sarcastic
- Train probe on clean + 4 training styles
- Evaluate on all 8 styles + clean

```bash
python experiments/experiment_1a_style_augmented_probes.py \
    --model google/gemma-3-4b-it \
    --model_size 4B \
    --questions datasets/4B/d_repe.json \
    --output_dir results_additional_experiments/exp1a
```


#### 1B: Per-Domain PCA Permutation Test

**Question:** Why does k*=0 (pooled) when multi-dim probes clearly help ?

**Design:**
- Run PCA + permutation null separately on D-RepE, D-Role, D-MASK
- Compare per-domain k* with pooled k*=0
- 1,000 permutations per test

```bash
python additional_experiments/experiment_1b_perdomain_pca.py \
    --model google/gemma-3-4b-it \
    --model_size 4B \
    --d_repe datasets/4B/d_repe.json \
    --d_role datasets/4B/d_role.json \
    --d_mask datasets/4B/d_mask.json \
    --existing_results aggregated_results/4B_model_all_results.json \
    --output_dir results_additional_experiments/exp1b
```



#### Verified vs Unverified Stratified Analysis (Study 1)

**Question:** Are near-perfect AUROC scores on D-RepE driven by genuine deception detection, or does the probe merely separate examples where the model followed the lie instruction from examples where it ignored it?

**Design:**
- Each D-RepE example carries a `verified` flag: `True` when the deceptive response was confirmed to omit the ground-truth answer (the model actually lied), `False` when it may have leaked the correct answer despite the lie instruction.
- The best-layer linear probe is evaluated separately on the verified-only and unverified-only test subsets.
- If AUROC remains high on the verified stratum, the probe detects genuine deception in the activations rather than surface instruction-following differences.

**How to run:**

The stratified analysis runs automatically as part of Study 1 — no separate script needed:

```bash
# Re-run Study 1 using cached activations and datasets
python scripts/run_all.py \
    --models 4B 27B \
    --studies 1 \
    --skip-datagen \
    --skip-activation
    --output_dir verified_vs_unverified_stratified
```

Results are saved to `outputs/results/study1/study1_results.json` under `scaling.<model_size>.stratified_auroc` and printed by `generate_tables.py`:

```bash
python scripts/generate_tables.py --results-dir ./outputs/results
```

Look for the **"Verified vs Unverified Stratified AUROC"** table in the output, which reports AUROC with 95% bootstrap CIs for both strata at each model size.

**Note on alignment:** The verified flags are read from the dataset JSON files and aligned with the cached activation arrays by replaying the same `RandomState(42)` permutation used during activation splitting — re-collecting activations is not required.

**Note on GPU:** On an A100 80 GB, run without quantization. If PyTorch reports `cuda.is_available() = False` despite a healthy `nvidia-smi`, the installed PyTorch was likely compiled against a newer CUDA than your driver supports. Fix with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

#### 1C: Cross-Domain Transfer at Target-Best Layers

**Question:** How much of off-diagonal failure is layer mismatch vs geometric disjointness?

**Design:** For each source→target pair, evaluate under three conditions:
1. Source probe at source's best layer (= existing Table 6)
2. Source probe weights at target's best layer (pure layer mismatch test)
3. Source data retrained at target's best layer (isolates geometric disjointness)

**Decomposition:**
- `layer_mismatch_effect` = Condition 2 - Condition 1
- `feature_relearning_effect` = Condition 3 - Condition 2
- `remaining_gap` = in-domain AUROC - Condition 3

```bash
 python additional_experiments/experiment_1c_target_layer_transfer.py \
    --model google/gemma-3-12b-it \
    --model_size 12B \
    --d_repe datasets/12B/d_repe.json \
    --d_role datasets/12B/d_role.json \
    --d_mask datasets/12B/d_mask.json \
    --existing_results aggregated_results/12B_model_all_results.json \
    --output_dir results_additional_experiments/exp1c
	
python additional_experiments/experiment_1c_target_layer_transfer.py \
    --model google/gemma-3-27b-it \
    --model_size 27B \
    --d_repe datasets/27B/d_repe.json \
    --d_role datasets/27B/d_role.json \
    --d_mask datasets/27B/d_mask.json \
    --existing_results aggregated_results/27B_model_all_results.json \
    --output_dir results_additional_experiments/exp1c
```


## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kumar-2026-pressure,
    title = "Pressure-Testing Deception Probes in {LLM}s: Scaling, Robustness, and the Geometry of Deceptive Representations",
    author = "Kumar, Sachin",
    editor = "Mille, Simon  and
      Gehrmann, Sebastian  and
      Schmidtov{\'a}, Patr{\'i}cia  and
      Du{\v{s}}ek, Ond{\v{r}}ej  and
      Fadaee, Marzieh  and
      Lo, Kyle  and
      Santus, Enrico  and
      Stanovsky, Gabriel",
    booktitle = "Proceedings of the Fifth Workshop on Generation, Evaluation and Metrics ({GEM})",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.gem-main.43/",
    doi = "10.18653/v1/2026.gem-main.43",
    pages = "472--489",
    ISBN = "979-8-89176-423-1",
}
```

Published in the [ACL Anthology](https://aclanthology.org/2026.gem-main.43/) (GEM Workshop @ ACL 2026, pages 472–489).
