"""
Experiment 1B: Per-Domain PCA Permutation Test
===============================================
Resolves the tension between k*=0 (pooled PCA) and multi-dim probes clearly
outperforming 1D probes (Table 5). Hypothesis: pooling geometrically distinct
deception types destroys per-domain structure.

Design:
  - Run PCA permutation test SEPARATELY on D-RepE, D-Role, D-MASK difference vectors
  - Compare per-domain k* with the pooled k*=0
  - If per-domain k*>0: "deception is multi-dimensional within each domain,
    but domains don't share a common subspace"
  - If per-domain k*=0 too: the multi-dim advantage is from non-linear structure,
    not a structured linear subspace
"""

import argparse
import json
import os
import logging
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_PERMUTATIONS = 1000
MAX_COMPONENTS = 20


def get_n_layers(model) -> int:
    """Get number of hidden layers, handling Gemma 3's nested multimodal config."""
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers
    raise AttributeError(f"Cannot find num_hidden_layers in config: {type(config)}")


def load_dataset(path: str) -> Dict:
    """
    Load a dataset JSON file.
    Expected format: {
        "train": [ { "idx", "question", "honest_response", "deceptive_response", ... }, ... ],
        "val": [...],
        "test": [...]
    }
    The honest_response/deceptive_response fields contain the full prompt+response text.
    """
    with open(path) as f:
        data = json.load(f)
    total = sum(len(data[split]) for split in data)
    logger.info(f"Loaded {path}: {total} items (train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])})")
    return data


def collect_activation_at_layer(
    model,
    tokenizer,
    text: str,
    layer: int,
    device: str,
) -> np.ndarray:
    """
    Feed a full prompt+response text through the model and collect
    the residual stream activation at the specified layer at the last token.
    Returns: (d_model,) array.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, d_model)
    # layer+1 because index 0 is the embedding layer
    h = outputs.hidden_states[layer + 1][0, -1, :].cpu().float().numpy()
    return h


def collect_difference_vectors(
    model,
    tokenizer,
    dataset_items: List[Dict],
    layer: int,
    device: str,
    dataset_name: str,
) -> np.ndarray:
    """
    For each paired item, compute dh = h_deceptive - h_honest at the given layer.
    Returns: (N, d_model) array of difference vectors.
    """
    diff_vectors = []
    n = len(dataset_items)

    for i, item in enumerate(dataset_items):
        h_honest = collect_activation_at_layer(
            model, tokenizer, item["honest_response"], layer, device
        )
        h_deceptive = collect_activation_at_layer(
            model, tokenizer, item["deceptive_response"], layer, device
        )
        diff_vectors.append(h_deceptive - h_honest)

        if (i + 1) % 50 == 0:
            logger.info(f"  [{dataset_name}] Collected {i+1}/{n} difference vectors")

    result = np.stack(diff_vectors)
    logger.info(f"  [{dataset_name}] Final shape: {result.shape}")
    return result


def permutation_null_eigenvalues(
    diff_vectors: np.ndarray,
    n_permutations: int,
    max_components: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute null distribution of eigenvalues by randomly flipping signs
    of difference vectors (equivalent to permuting honest/deceptive labels).
    Returns: (max_components, n_permutations) array of null eigenvalues.
    """
    rng = np.random.default_rng(seed)
    n, d = diff_vectors.shape
    k = min(max_components, n - 1, d)

    null_eigenvalues = np.zeros((k, n_permutations))

    for perm_i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted = diff_vectors * signs[:, np.newaxis]
        pca = PCA(n_components=k)
        pca.fit(permuted)
        null_eigenvalues[:, perm_i] = pca.explained_variance_

        if (perm_i + 1) % 200 == 0:
            logger.info(f"    Permutation {perm_i+1}/{n_permutations}")

    return null_eigenvalues


def per_domain_pca_test(
    diff_vectors: np.ndarray,
    domain_name: str,
    n_permutations: int,
    max_components: int,
    alpha: float = 0.05,
) -> Dict:
    """
    PCA with permutation null on a single domain's difference vectors.
    """
    n, d = diff_vectors.shape
    k = min(max_components, n - 1, d)
    logger.info(f"  PCA on {domain_name}: {n} vectors, d={d}, testing k={k} components")

    # Real PCA
    pca = PCA(n_components=k)
    pca.fit(diff_vectors)
    real_eigenvalues = pca.explained_variance_
    real_var_explained = pca.explained_variance_ratio_

    # Permutation null
    logger.info(f"    Running {n_permutations} permutations...")
    null_eigenvalues = permutation_null_eigenvalues(diff_vectors, n_permutations, k)
    null_95th = np.percentile(null_eigenvalues, 100 * (1 - alpha), axis=1)

    # k*: number of leading components exceeding null (stop at first failure)
    k_star = 0
    for i in range(k):
        if real_eigenvalues[i] > null_95th[i]:
            k_star = i + 1
        else:
            break

    # Effect size: ratio of real eigenvalue to null 95th percentile
    effect_ratios = real_eigenvalues[:k] / null_95th[:k]

    result = {
        "domain": domain_name,
        "n_vectors": int(n),
        "d_model": int(d),
        "k_star": int(k_star),
        "real_eigenvalues": real_eigenvalues.tolist(),
        "real_var_explained": real_var_explained.tolist(),
        "null_95th_percentile": null_95th.tolist(),
        "effect_ratios": effect_ratios.tolist(),
        "cumulative_var_explained": np.cumsum(real_var_explained).tolist(),
    }

    logger.info(
        f"    k*={k_star} | PC1 var={real_var_explained[0]:.3f} "
        f"| PC1 ratio={effect_ratios[0]:.3f} "
        f"| PC1 {'EXCEEDS' if k_star >= 1 else 'below'} null"
    )
    if k_star > 0:
        for i in range(k_star):
            logger.info(
                f"      PC{i+1}: var={real_var_explained[i]:.4f}, "
                f"eigenval={real_eigenvalues[i]:.1f}, "
                f"null_95={null_95th[i]:.1f}, "
                f"ratio={effect_ratios[i]:.3f}"
            )

    return result


def run_experiment(args):
    """Main experiment."""
    logger.info(f"=== Experiment 1B: Per-Domain PCA Permutation Test ({args.model_size}) ===")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load datasets ──
    datasets = {
        "repe": load_dataset(args.d_repe),
        "role": load_dataset(args.d_role),
        "mask": load_dataset(args.d_mask),
    }

    # ── Load existing results for best layers per domain ──
    if args.existing_results and os.path.exists(args.existing_results):
        with open(args.existing_results) as f:
            existing = json.load(f)
        probe_layers = {
            ds: existing["study2"]["transfer"]["probes"][ds]["layer"]
            for ds in ["repe", "role", "mask"]
        }
        logger.info(f"Best layers from existing results: {probe_layers}")
    else:
        logger.error("--existing_results required (provides per-domain best layers)")
        return

    # ── Load model ──
    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = str(next(model.parameters()).device)
    n_layers = get_n_layers(model)
    logger.info(f"Model loaded: {n_layers} layers, device={device}")

    # ── Collect difference vectors per domain ──
    # Use train+val splits (test held out, matching main experiment protocol)
    domain_diff_vectors = {}
    for ds_name in ["repe", "role", "mask"]:
        items = datasets[ds_name]["train"] + datasets[ds_name]["val"]
        layer = probe_layers[ds_name]
        logger.info(f"Collecting {ds_name} dh at layer {layer} ({len(items)} pairs)...")

        domain_diff_vectors[ds_name] = collect_difference_vectors(
            model, tokenizer, items, layer, device, ds_name
        )

    # Free GPU memory before CPU-heavy permutation tests
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, running permutation tests on CPU...")

    # ── Run per-domain PCA permutation tests ──
    results = {
        "model": args.model,
        "model_size": args.model_size,
        "n_layers": n_layers,
        "probe_layers": probe_layers,
        "n_permutations": args.n_permutations,
        "per_domain": {},
        "pooled": None,
    }

    for ds_name, dv in domain_diff_vectors.items():
        logger.info(f"Testing {ds_name} (n={dv.shape[0]}, d={dv.shape[1]})...")
        results["per_domain"][ds_name] = per_domain_pca_test(
            dv, ds_name, args.n_permutations, MAX_COMPONENTS
        )

    # ── Pooled test for comparison ──
    logger.info("Running pooled PCA permutation test...")
    pooled_dv = np.concatenate(list(domain_diff_vectors.values()), axis=0)
    results["pooled"] = per_domain_pca_test(
        pooled_dv, "pooled", args.n_permutations, MAX_COMPONENTS
    )

    # ── Summary ──
    domain_ks = {ds: results["per_domain"][ds]["k_star"] for ds in ["repe", "role", "mask"]}
    pooled_k = results["pooled"]["k_star"]

    results["summary"] = {
        "per_domain_k_stars": domain_ks,
        "pooled_k_star": pooled_k,
        "any_domain_positive": any(k > 0 for k in domain_ks.values()),
        "interpretation": _interpret(domain_ks, pooled_k),
    }

    # ── Save ──
    out_path = os.path.join(args.output_dir, f"exp1b_{args.model_size}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    _print_summary(results)
    return results


def _interpret(domain_ks: Dict[str, int], pooled_k: int) -> str:
    any_positive = any(k > 0 for k in domain_ks.values())
    positive_domains = [ds for ds, k in domain_ks.items() if k > 0]

    if any_positive and pooled_k == 0:
        return (
            f"Per-domain k*>0 for {positive_domains} but pooled k*=0. "
            "Deception is multi-dimensional within individual domains, but domains don't share "
            "a common subspace. Pooling geometrically distinct directions destroys per-domain "
            "structure. This resolves the Table 5 tension and supports domain-specific H-SUB."
        )
    elif not any_positive and pooled_k == 0:
        return (
            "k*=0 for ALL domains AND pooled. The multi-dimensional probe advantage (Table 5) "
            "arises from non-linear structure or distributed weak signals that individually "
            "fall below the permutation null. Even within domains, deception does not form "
            "a statistically significant linear subspace."
        )
    elif any_positive and pooled_k > 0:
        return (
            f"k*>0 for both per-domain ({positive_domains}) and pooled (k*={pooled_k}). "
            "A shared deception subspace may exist. This would strengthen H-SUB support."
        )
    elif not any_positive and pooled_k > 0:
        return (
            f"Unexpected: no per-domain structure but pooled k*={pooled_k}. "
            "Pooling may create an artifact."
        )
    return f"Unhandled case: domain_ks={domain_ks}, pooled_k={pooled_k}"


def _print_summary(results: dict):
    print("\n" + "=" * 75)
    print(f"EXPERIMENT 1B SUMMARY — {results['model_size']}")
    print("=" * 75)
    print(f"  Layers used: {results['probe_layers']}")
    print(f"  Permutations: {results['n_permutations']}")
    print()
    print(f"  {'Domain':<12} {'n':<6} {'k*':<5} {'PC1 Var%':<10} {'PC1 Ratio':<12} {'PC2 Var%':<10} {'PC3 Var%':<10}")
    print("  " + "-" * 65)

    for ds_name in ["repe", "role", "mask"]:
        res = results["per_domain"][ds_name]
        label = f"D-{ds_name.upper()}"
        _print_row(label, res)

    print("  " + "-" * 65)
    _print_row("POOLED", results["pooled"])

    print()
    s = results["summary"]
    print(f"  Per-domain k*: RepE={s['per_domain_k_stars']['repe']}, "
          f"Role={s['per_domain_k_stars']['role']}, "
          f"MASK={s['per_domain_k_stars']['mask']}")
    print(f"  Pooled k*:     {s['pooled_k_star']}")
    print(f"\n  Interpretation: {s['interpretation']}")
    print("=" * 75)


def _print_row(label: str, res: dict):
    pc1 = res["real_var_explained"][0] * 100
    pc2 = res["real_var_explained"][1] * 100 if len(res["real_var_explained"]) > 1 else 0
    pc3 = res["real_var_explained"][2] * 100 if len(res["real_var_explained"]) > 2 else 0
    ratio1 = res["effect_ratios"][0]
    marker = " *" if res["k_star"] > 0 else ""
    print(
        f"  {label:<12} {res['n_vectors']:<6} {res['k_star']:<5}"
        f"{pc1:<10.1f} {ratio1:<12.3f} {pc2:<10.1f} {pc3:<10.1f}{marker}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1B: Per-Domain PCA Permutation Test")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--model_size", type=str, required=True, choices=["1B", "4B", "12B", "27B"])
    parser.add_argument("--d_repe", type=str, required=True, help="Path to d_repe.json")
    parser.add_argument("--d_role", type=str, required=True, help="Path to d_role.json")
    parser.add_argument("--d_mask", type=str, required=True, help="Path to d_mask.json")
    parser.add_argument("--existing_results", type=str, required=True,
                        help="Path to {model_size}_model_all_results.json (for best layers)")
    parser.add_argument("--output_dir", type=str, default="results/exp1b")
    parser.add_argument("--n_permutations", type=int, default=1000)
    args = parser.parse_args()
    run_experiment(args)
