# llm-deception-probe-stress-test

Full implementation of the experimental pipeline for the CoNLL 2026 paper submission:
*"Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations"*

## Overview

This codebase implements four studies testing competing hypotheses about how deception is encoded in LLM activations:

| Study | Section | What it tests |
|-------|---------|---------------|
| **Study 1** | §5 | Scaling laws: emergence of linear separability across 1B→27B |
| **Study 2** | §6 | Geometry: is deception a direction, subspace, or cone? |
| **Study 3** | §7 | Adversarial robustness: five-probe battery against RL-obfuscation |
| **Study 4** | §8 | Entropy proxy: are probes detecting cognitive load? |


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