"""
Experiment 1C: Cross-Domain Transfer at Target-Best Layers
==========================================================
Decomposes off-diagonal transfer failures into two components:
  (a) Layer mismatch — the source probe was optimized at the wrong layer
  (b) Genuine geometric disjointness — deception types occupy different directions

Design:
  For each (source_dataset, target_dataset) pair:
  1. Evaluate source probe at SOURCE best layer (existing Table 6)
  2. Evaluate source probe at TARGET best layer (new)
  3. Retrain source probe at TARGET best layer (new — controls for layer-specific features)

  If transfer improves at target layer → layer mismatch explains some failure
  If transfer remains poor → genuine geometric disjointness (H-LIN rejection is rock-solid)
"""

import argparse
import json
import os
import logging
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["repe", "role", "mask"]
DATASET_LABELS = {"repe": "D-RepE", "role": "D-Role", "mask": "D-MASK"}
BOOTSTRAP_N = 10_000


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
    Load dataset JSON. Expected format:
    { "train": [{idx, question, honest_response, deceptive_response, ...}], "val": [...], "test": [...] }
    """
    with open(path) as f:
        data = json.load(f)
    total = sum(len(data[s]) for s in data)
    logger.info(f"Loaded {path}: {total} items (train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])})")
    return data


def collect_activations_at_layers(
    model,
    tokenizer,
    items: List[Dict],
    layers: List[int],
    device: str,
    dataset_name: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    For each item, feed honest_response and deceptive_response through the model,
    collect last-token activations at each specified layer.

    Returns:
        X_by_layer: dict mapping layer -> (N, d_model) array
                    where N = 2 * len(items), first half honest, second half deceptive
        y: (N,) label array (0=honest, 1=deceptive)
    """
    # Collect per-layer activations
    layer_acts = {l: [] for l in layers}  # layer -> list of (d_model,) arrays
    labels = []
    n = len(items)

    for label_val, key in [(0, "honest_response"), (1, "deceptive_response")]:
        label_name = "honest" if label_val == 0 else "deceptive"
        for i, item in enumerate(items):
            text = item[key]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in layers:
                h = outputs.hidden_states[l + 1][0, -1, :].cpu().float().numpy()
                layer_acts[l].append(h)

            labels.append(label_val)

            if (i + 1) % 100 == 0:
                logger.info(f"  [{dataset_name}/{label_name}] {i+1}/{n}")

    X_by_layer = {l: np.stack(layer_acts[l]) for l in layers}
    y = np.array(labels)

    logger.info(f"  [{dataset_name}] Collected: {X_by_layer[layers[0]].shape[0]} samples at {len(layers)} layers")
    return X_by_layer, y


def train_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegressionCV:
    """Train L2-regularized logistic regression probe."""
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        scoring="roc_auc",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def evaluate_with_bootstrap(
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """Evaluate probe with bootstrap CI."""
    y_score = clf.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, y_score)

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try:
            boot.append(roc_auc_score(y[idx], y_score[idx]))
        except ValueError:
            boot.append(0.5)

    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return auroc, ci


def run_experiment(args):
    """Main experiment logic."""
    logger.info(f"=== Experiment 1C: Target-Layer Transfer ({args.model_size}) ===")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load existing results for best layers and baseline transfer matrix ──
    with open(args.existing_results) as f:
        existing = json.load(f)

    best_layers = {
        ds: existing["study2"]["transfer"]["probes"][ds]["layer"]
        for ds in DATASETS
    }
    existing_matrix = existing["study2"]["transfer"]["matrix"]
    logger.info(f"Best layers per domain: {best_layers}")

    # ── Load datasets ──
    dataset_paths = {"repe": args.d_repe, "role": args.d_role, "mask": args.d_mask}
    datasets = {ds: load_dataset(path) for ds, path in dataset_paths.items()}

    # Unique layers we need activations for
    unique_layers = sorted(set(best_layers.values()))
    logger.info(f"Unique layers to collect: {unique_layers}")

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

    # ── Collect activations for all datasets at all needed layers ──
    # Use train+val for training probes, test for evaluation
    all_data = {}  # ds_name -> {"train": {layer: X, y}, "test": {layer: X, y}}

    for ds_name in DATASETS:
        train_items = datasets[ds_name]["train"] + datasets[ds_name]["val"]
        test_items = datasets[ds_name]["test"]

        logger.info(f"Collecting {ds_name} train activations ({len(train_items)} items, {len(unique_layers)} layers)...")
        train_X_by_layer, train_y = collect_activations_at_layers(
            model, tokenizer, train_items, unique_layers, device, f"{ds_name}/train"
        )

        logger.info(f"Collecting {ds_name} test activations ({len(test_items)} items)...")
        test_X_by_layer, test_y = collect_activations_at_layers(
            model, tokenizer, test_items, unique_layers, device, f"{ds_name}/test"
        )

        all_data[ds_name] = {
            "train_X": train_X_by_layer,
            "train_y": train_y,
            "test_X": test_X_by_layer,
            "test_y": test_y,
        }

    # Free GPU
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded. Running probe training and evaluation on CPU...")

    # ── Build transfer matrices under three conditions ──
    results = {
        "model": args.model,
        "model_size": args.model_size,
        "n_layers": n_layers,
        "best_layers": best_layers,
        "transfer_matrices": {
            "source_layer": {},
            "target_layer": {},
            "retrained_target": {},
        },
        "decomposition": {},
    }

    for source_ds in DATASETS:
        src_layer = best_layers[source_ds]

        for target_ds in DATASETS:
            tgt_layer = best_layers[target_ds]
            key = f"{source_ds}_to_{target_ds}"

            src_train_X = all_data[source_ds]["train_X"]
            src_train_y = all_data[source_ds]["train_y"]
            tgt_test_X = all_data[target_ds]["test_X"]
            tgt_test_y = all_data[target_ds]["test_y"]

            # ── Condition 1: Source probe trained & evaluated at source layer ──
            probe_src = train_probe(src_train_X[src_layer], src_train_y)
            auroc_src, ci_src = evaluate_with_bootstrap(
                probe_src, tgt_test_X[src_layer], tgt_test_y
            )

            results["transfer_matrices"]["source_layer"][key] = {
                "auroc": float(auroc_src),
                "auroc_ci": ci_src,
                "probe_trained_at_layer": src_layer,
                "evaluated_at_layer": src_layer,
            }

            # ── Condition 2: Same probe weights, evaluated at target layer ──
            # (same weights learned at src_layer, applied to activations from tgt_layer)
            auroc_tgt_same, ci_tgt_same = evaluate_with_bootstrap(
                probe_src, tgt_test_X[tgt_layer], tgt_test_y
            )

            results["transfer_matrices"]["target_layer"][key] = {
                "auroc": float(auroc_tgt_same),
                "auroc_ci": ci_tgt_same,
                "probe_trained_at_layer": src_layer,
                "evaluated_at_layer": tgt_layer,
            }

            # ── Condition 3: Retrain source data at target layer ──
            probe_retrained = train_probe(src_train_X[tgt_layer], src_train_y)
            auroc_retrained, ci_retrained = evaluate_with_bootstrap(
                probe_retrained, tgt_test_X[tgt_layer], tgt_test_y
            )

            results["transfer_matrices"]["retrained_target"][key] = {
                "auroc": float(auroc_retrained),
                "auroc_ci": ci_retrained,
                "probe_trained_at_layer": tgt_layer,
                "evaluated_at_layer": tgt_layer,
            }

            # ── Decomposition ──
            layer_mismatch = auroc_tgt_same - auroc_src
            feature_relearning = auroc_retrained - auroc_tgt_same
            total_improvement = auroc_retrained - auroc_src

            # Remaining gap from in-domain performance
            in_domain = existing_matrix.get(target_ds, {}).get(target_ds)
            remaining_gap = (in_domain - auroc_retrained) if in_domain is not None else None

            results["decomposition"][key] = {
                "source_layer_auroc": float(auroc_src),
                "target_layer_same_probe_auroc": float(auroc_tgt_same),
                "retrained_target_layer_auroc": float(auroc_retrained),
                "layer_mismatch_effect": float(layer_mismatch),
                "feature_relearning_effect": float(feature_relearning),
                "total_improvement": float(total_improvement),
                "remaining_gap_to_in_domain": float(remaining_gap) if remaining_gap is not None else None,
                "is_diagonal": source_ds == target_ds,
                "source_layer": src_layer,
                "target_layer": tgt_layer,
            }

            is_diag = "DIAG" if source_ds == target_ds else ""
            logger.info(
                f"  {DATASET_LABELS[source_ds]:>8} -> {DATASET_LABELS[target_ds]:<8} "
                f"| C1(src_layer)={auroc_src:.3f} "
                f"| C2(tgt_layer,same_wt)={auroc_tgt_same:.3f} "
                f"| C3(retrained_tgt)={auroc_retrained:.3f} "
                f"| total_delta={total_improvement:+.3f} {is_diag}"
            )

    # ── Summary ──
    off_diag = [v for v in results["decomposition"].values() if not v["is_diagonal"]]
    mean_src = float(np.mean([v["source_layer_auroc"] for v in off_diag]))
    mean_retrained = float(np.mean([v["retrained_target_layer_auroc"] for v in off_diag]))
    mean_improvement = float(np.mean([v["total_improvement"] for v in off_diag]))
    mean_layer_mismatch = float(np.mean([v["layer_mismatch_effect"] for v in off_diag]))
    mean_feature_relearn = float(np.mean([v["feature_relearning_effect"] for v in off_diag]))
    gaps = [v["remaining_gap_to_in_domain"] for v in off_diag if v["remaining_gap_to_in_domain"] is not None]
    mean_remaining_gap = float(np.mean(gaps)) if gaps else None

    results["summary"] = {
        "n_off_diagonal_pairs": len(off_diag),
        "mean_off_diagonal_source_layer": mean_src,
        "mean_off_diagonal_retrained_target": mean_retrained,
        "mean_total_improvement": mean_improvement,
        "mean_layer_mismatch_effect": mean_layer_mismatch,
        "mean_feature_relearning_effect": mean_feature_relearn,
        "mean_remaining_gap": mean_remaining_gap,
        "interpretation": _interpret(mean_src, mean_retrained, mean_improvement),
    }

    # ── Save ──
    out_path = os.path.join(args.output_dir, f"exp1c_{args.model_size}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    _print_summary(results)
    return results


def _interpret(mean_src: float, mean_retrained: float, mean_improvement: float) -> str:
    if mean_improvement < 0.05:
        return (
            f"Minimal improvement from target-layer evaluation (mean delta={mean_improvement:+.3f}). "
            "Layer mismatch is NOT the primary cause of transfer failure. "
            "H-LIN rejection is rock-solid: deception types are genuinely geometrically disjoint."
        )
    elif mean_improvement > 0.15 and mean_retrained > 0.75:
        return (
            f"Substantial improvement (mean delta={mean_improvement:+.3f}, retrained={mean_retrained:.3f}). "
            "Layer mismatch explains a significant portion of transfer failure. "
            "H-LIN rejection should be softened — some cross-domain signal exists at the right layer."
        )
    elif 0.05 <= mean_improvement <= 0.15:
        return (
            f"Moderate improvement (mean delta={mean_improvement:+.3f}). "
            "Layer mismatch contributes but does not fully explain transfer failure. "
            "Both layer specificity and geometric disjointness play a role."
        )
    else:
        return (
            f"Mixed: improvement={mean_improvement:+.3f}, retrained={mean_retrained:.3f}. "
            "Layer mismatch helps but probes remain substantially below in-domain performance."
        )


def _print_summary(results: dict):
    print("\n" + "=" * 100)
    print(f"EXPERIMENT 1C SUMMARY — {results['model_size']}")
    print("=" * 100)
    print(f"  Best layers: {results['best_layers']}")
    print()
    print(
        f"  {'Transfer':<22} {'Src Lyr':<8} {'Tgt Lyr':<8} "
        f"{'C1:Src':<10} {'C2:Tgt(same)':<13} {'C3:Retrained':<13} "
        f"{'Layer d':<10} {'Relearn d':<10} {'Total d':<10}"
    )
    print("  " + "-" * 95)

    for key, dec in results["decomposition"].items():
        src, tgt = key.split("_to_")
        marker = " *" if dec["is_diagonal"] else ""
        print(
            f"  {DATASET_LABELS[src]}->{DATASET_LABELS[tgt]:<8}{marker:<3}"
            f"{dec['source_layer']:<8}{dec['target_layer']:<8}"
            f"{dec['source_layer_auroc']:<10.3f}"
            f"{dec['target_layer_same_probe_auroc']:<13.3f}"
            f"{dec['retrained_target_layer_auroc']:<13.3f}"
            f"{dec['layer_mismatch_effect']:<+10.3f}"
            f"{dec['feature_relearning_effect']:<+10.3f}"
            f"{dec['total_improvement']:<+10.3f}"
        )

    print("  " + "-" * 95)
    s = results["summary"]
    print(f"  Off-diagonal means:")
    print(f"    C1 (source layer):         {s['mean_off_diagonal_source_layer']:.3f}")
    print(f"    C3 (retrained at target):  {s['mean_off_diagonal_retrained_target']:.3f}")
    print(f"    Layer mismatch effect:     {s['mean_layer_mismatch_effect']:+.3f}")
    print(f"    Feature relearning effect: {s['mean_feature_relearning_effect']:+.3f}")
    print(f"    Total improvement:         {s['mean_total_improvement']:+.3f}")
    if s["mean_remaining_gap"] is not None:
        print(f"    Remaining gap to in-domain: {s['mean_remaining_gap']:.3f}")
    print(f"\n  Interpretation: {s['interpretation']}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1C: Target-Layer Transfer Decomposition")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--model_size", type=str, required=True, choices=["1B", "4B", "12B", "27B"])
    parser.add_argument("--d_repe", type=str, required=True, help="Path to d_repe.json")
    parser.add_argument("--d_role", type=str, required=True, help="Path to d_role.json")
    parser.add_argument("--d_mask", type=str, required=True, help="Path to d_mask.json")
    parser.add_argument("--existing_results", type=str, required=True,
                        help="Path to {model_size}_model_all_results.json")
    parser.add_argument("--output_dir", type=str, default="results/exp1c")
    args = parser.parse_args()
    run_experiment(args)
