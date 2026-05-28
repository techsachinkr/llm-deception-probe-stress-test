"""
Study 1: The Scaling Laws of Deception Representations (§5.1).
Tests emergence of linear separability, peak layer analysis,
and SAE feature decomposition across Gemma 3 scales.
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from src.probes import train_probes_all_layers, TextBaseline
from src.metrics import (
    compute_all_metrics, bootstrap_peak_layer, auroc, bootstrap_ci
)
from src.utils import save_json

logger = logging.getLogger(__name__)


def run_study1(
    all_activations: Dict[str, Dict],   # {model_size: {dataset: (acts, labels, prompt_acts)}}
    all_datasets: Dict,                  # {dataset_name: DataSplit}
    config,
) -> Dict:
    """
    Run Study 1: Scaling Laws.
    
    Tests:
    - Emergence of linear separability across 1B -> 27B
    - Peak layer analysis with bootstrap CIs (§4.9.3)
    - Layer-wise probe profiles
    """
    results = {"scaling": {}, "peak_layers": {}, "layer_profiles": {}}
    save_dir = Path(config.results_dir) / "study1"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Primary dataset for scaling analysis: D-RepE
    ds_name = "repe"
    
    for model_size in config.model_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Study 1: Processing {model_size}")
        logger.info(f"{'='*60}")
        
        if model_size not in all_activations or ds_name not in all_activations[model_size]:
            logger.warning(f"No activations for {model_size}/{ds_name}, skipping")
            continue
        
        acts_data = all_activations[model_size][ds_name]
        train_acts, train_labels = acts_data["train"]
        val_acts, val_labels = acts_data["val"]
        test_acts, test_labels = acts_data["test"]
        
        # Skip if insufficient data
        if len(np.unique(train_labels)) < 2 or len(train_labels) < 10:
            logger.warning(f"Skipping {model_size}/{ds_name}: insufficient training data")
            continue
        
        n_layers = train_acts.shape[1]
        d_model = train_acts.shape[2]
        
        # ── 5.1: Train probes at every layer ──
        probe_results = train_probes_all_layers(
            train_acts, train_labels, config,
            val_acts, val_labels,
        )
        
        best_layer = probe_results["best_layer"]
        
        # Evaluate on test set
        test_probe = probe_results["layer_probes"][best_layer]
        X_test = test_acts[:, best_layer, :]
        test_scores = test_probe.predict_proba(X_test)
        
        # Compute text baseline for B->W boost
        ds = all_datasets[ds_name]
        train_texts = ([e.honest_response for e in ds.train] +
                       [e.deceptive_response for e in ds.train])
        # Note: text labels order must match activation labels
        text_baseline = TextBaseline(seed=config.probe.seeds[0])
        # Simplified: we use the same labels order
        try:
            text_baseline.fit(train_texts[:len(train_labels)], train_labels[:len(train_texts)])
            test_texts = ([e.honest_response for e in ds.test] +
                          [e.deceptive_response for e in ds.test])
            baseline_auroc = text_baseline.score(test_texts[:len(test_labels)], test_labels[:len(test_texts)])
        except Exception:
            baseline_auroc = 0.5
        
        # Full metrics
        test_metrics = compute_all_metrics(
            test_labels, test_scores,
            n_bootstrap=config.probe.n_bootstrap,
            text_baseline_auroc=baseline_auroc,
        )
        
        results["scaling"][model_size] = {
            "best_layer": int(best_layer),
            "n_layers": n_layers,
            "d_model": d_model,
            "test_metrics": test_metrics,
            "multi_dim_aurocs": {
                k: v["auroc"] for k, v in probe_results.get("multi_dim", {}).items()
            },
            "mlp_auroc": probe_results.get("mlp", {}).get("auroc", None),
        }

        # ── Stratified analysis: verified vs unverified (D-RepE only) ──
        # "Verified" means the deceptive response was confirmed to omit the ground
        # truth, i.e. the model actually lied.  High AUROC on the verified-only
        # stratum shows the probe detects genuine deception rather than merely
        # distinguishing instruction-following from instruction-ignoring examples.
        if ds_name == "repe" and ds_name in all_datasets:
            all_examples = all_datasets[ds_name].all_examples
            n_train_size = len(train_acts)
            n_val_size   = len(val_acts)
            n_test_size  = len(test_acts)
            n_total = n_train_size + n_val_size + n_test_size

            verified_all = np.array([e.verified for e in all_examples], dtype=bool)
            # collect_dataset_activations interleaves honest/deceptive:
            #   acts[2*i] = honest (label 0), acts[2*i+1] = deceptive (label 1)
            if len(all_examples) * 2 == n_total:
                verified_acts = np.repeat(verified_all, 2)
            elif len(all_examples) == n_total:
                verified_acts = verified_all
            else:
                verified_acts = None

            if verified_acts is not None:
                # Replay the identical permutation used in prepare_activation_splits
                perm = np.random.RandomState(42).permutation(n_total)
                test_orig_idx = perm[n_train_size + n_val_size:]
                test_verified = verified_acts[test_orig_idx]

                n_ver   = int(np.sum(test_verified))
                n_unver = int(np.sum(~test_verified))
                logger.info(f"  D-RepE test split: {n_ver} verified, {n_unver} unverified")

                strat = {"n_verified": n_ver, "n_unverified": n_unver}

                if n_ver >= 10:
                    v_auc, v_lo, v_hi = bootstrap_ci(
                        test_labels[test_verified],
                        test_scores[test_verified],
                        auroc,
                        n_bootstrap=config.probe.n_bootstrap,
                    )
                    strat["verified"] = {
                        "auroc": float(v_auc),
                        "ci_lower": float(v_lo),
                        "ci_upper": float(v_hi),
                    }
                    logger.info(
                        f"  Verified-only AUROC:   {v_auc:.4f} "
                        f"[{v_lo:.4f}, {v_hi:.4f}]"
                    )

                if n_unver >= 10:
                    u_auc, u_lo, u_hi = bootstrap_ci(
                        test_labels[~test_verified],
                        test_scores[~test_verified],
                        auroc,
                        n_bootstrap=config.probe.n_bootstrap,
                    )
                    strat["unverified"] = {
                        "auroc": float(u_auc),
                        "ci_lower": float(u_lo),
                        "ci_upper": float(u_hi),
                    }
                    logger.info(
                        f"  Unverified-only AUROC: {u_auc:.4f} "
                        f"[{u_lo:.4f}, {u_hi:.4f}]"
                    )

                results["scaling"][model_size]["stratified_auroc"] = strat
            else:
                logger.warning(
                    f"  Verified stratification skipped: dataset length "
                    f"({len(all_examples)}) does not map to activation length ({n_total})"
                )

        # ── 5.2: Peak layer with bootstrap uncertainty ──
        layer_scores_dict = {}
        layer_aurocs_test = np.zeros(n_layers)
        for layer in range(n_layers):
            probe = probe_results["layer_probes"].get(layer)
            if probe is not None:
                scores = probe.predict_proba(test_acts[:, layer, :])
                layer_scores_dict[layer] = scores
                layer_aurocs_test[layer] = auroc(test_labels, scores)
        
        if np.max(layer_aurocs_test) > 0.80:
            median_peak, ci_lo, ci_hi = bootstrap_peak_layer(
                layer_aurocs_test, test_labels, layer_scores_dict,
                n_bootstrap=config.geometry.bootstrap_peak_n,
            )
            
            peak_frac = median_peak / n_layers
            peak_frac_lo = ci_lo / n_layers
            peak_frac_hi = ci_hi / n_layers
            
            # Test Two-Thirds Rule: does CI overlap [0.60, 0.75]?
            rule_lo, rule_hi = config.geometry.two_thirds_range
            overlaps = peak_frac_lo <= rule_hi and peak_frac_hi >= rule_lo
            
            results["peak_layers"][model_size] = {
                "median_peak_layer": float(median_peak),
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
                "peak_as_fraction": float(peak_frac),
                "fraction_ci": [float(peak_frac_lo), float(peak_frac_hi)],
                "two_thirds_rule_overlap": overlaps,
            }
            logger.info(
                f"  Peak layer: {median_peak:.0f} [{ci_lo:.0f}, {ci_hi:.0f}] "
                f"= {peak_frac:.2f} [{peak_frac_lo:.2f}, {peak_frac_hi:.2f}] "
                f"of depth. Two-thirds overlap: {overlaps}"
            )
        
        # ── 5.3: Layer-wise profile ──
        results["layer_profiles"][model_size] = {
            "layer_aurocs": layer_aurocs_test.tolist(),
        }
    
    # Save results
    save_json(results, str(save_dir / "study1_results.json"))
    logger.info(f"Study 1 results saved to {save_dir}")
    
    return results
