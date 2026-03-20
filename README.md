# llm-deception-probe-stress-test

Full implementation of the experimental pipeline for  paper submission titled ""Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations"*" into Generation, Evaluation, and Metrics 
(GEM) Workshop at ACL 2026.

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


