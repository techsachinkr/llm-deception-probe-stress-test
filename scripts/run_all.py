#!/usr/bin/env python3
"""
Main orchestrator for all experiments.
Runs Studies 1-4 from the paper:
  "Pressure-Testing Deception Probes in LLMs"

Usage:
    # Full pipeline (all models, all studies)
    python scripts/run_all.py

    # Quick test on 1B only
    python scripts/run_all.py --models 1B --quick

    # Specific studies
    python scripts/run_all.py --models 1B 4B --studies 1 2

    # Resume from cached activations
    python scripts/run_all.py --skip-datagen --skip-activation
    
    # With quantization for memory-constrained GPUs
    python scripts/run_all.py --load-in-4bit --models 12B 27B
"""
import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from configs.config import ExperimentConfig, MODELS, load_config
from src.utils import setup_logging, set_seed, load_model_and_tokenizer, save_json, get_gpu_memory
from src.data_generation_fast import generate_all_datasets, DataSplit
from src.activation_collector import ActivationCollector, collect_dataset_activations
from src.probes import LinearProbe, train_probes_all_layers
from src.study1_scaling import run_study1
from src.study2_geometry import run_study2
from src.study3_adversarial import run_study3
from src.study4_entropy import run_study4
from src.sae_analysis import run_sae_analysis

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run deception probe experiments")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config override")
    parser.add_argument("--models", nargs="+", default=None, help="Model sizes to run (e.g., 1B 4B 12B 27B)")
    parser.add_argument("--studies", nargs="+", type=int, default=None, help="Which studies to run (1-4)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: smaller datasets, fewer bootstraps")
    parser.add_argument("--skip-datagen", action="store_true", help="Skip dataset generation (use cached)")
    parser.add_argument("--skip-activation", action="store_true", help="Skip activation collection (use cached)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load models in 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load models in 8-bit quantization")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def prepare_activation_splits(
    activations: np.ndarray,  # (n, n_layers, d_model)
    labels: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Split activations into train/val/test matching dataset splits."""
    n = len(labels)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    # Shuffle deterministically
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    
    acts_shuffled = activations[idx]
    labels_shuffled = labels[idx]
    
    return {
        "train": (acts_shuffled[:n_train], labels_shuffled[:n_train]),
        "val": (acts_shuffled[n_train:n_train+n_val], labels_shuffled[n_train:n_train+n_val]),
        "test": (acts_shuffled[n_train+n_val:], labels_shuffled[n_train+n_val:]),
    }


def main():
    args = parse_args()
    setup_logging("INFO")
    
    # ── Configuration ──
    config = load_config(args.config)
    
    if args.models:
        config.model_sizes = args.models
    if args.studies:
        config.run_study1 = 1 in args.studies
        config.run_study2 = 2 in args.studies
        config.run_study3 = 3 in args.studies
        config.run_study4 = 4 in args.studies
    if args.load_in_4bit:
        config.load_in_4bit = True
    if args.load_in_8bit:
        config.load_in_8bit = True
    if args.output_dir:
        config.output_dir = args.output_dir
        config.results_dir = os.path.join(args.output_dir, "results")
        config.activation_dir = os.path.join(args.output_dir, "activations")
    if args.quick:
        config.data.n_per_class = 100
        config.probe.n_bootstrap = 1000
        config.geometry.n_permutations = 100
        config.geometry.bootstrap_peak_n = 1000
        config.rl.n_steps = 100
        config.distractor.n_per_style_per_class = 20
        logger.info("Quick mode enabled: reduced dataset sizes and iterations")
    
    config.setup_dirs()
    set_seed(args.seed)
    
    logger.info("="*70)
    logger.info("  PRESSURE-TESTING DECEPTION PROBES — EXPERIMENTAL PIPELINE")
    logger.info("="*70)
    logger.info(f"Models: {config.model_sizes}")
    logger.info(f"Studies: {[i for i in [1,2,3,4] if getattr(config, f'run_study{i}')]}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {config.output_dir}")
    if torch.cuda.is_available():
        get_gpu_memory()
    
    # Save config for reproducibility
    config.save(os.path.join(config.output_dir, "config.json"))
    
    all_activations = {}  # {model_size: {dataset: {"train": (acts, labels), "val": ..., "test": ...}}}
    all_datasets = {}
    blue_team_probes = {}  # {model_size: (probe, layer)}
    
    start_time = time.time()
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Data Generation & Activation Collection
    # ══════════════════════════════════════════════════════════════
    for model_size in config.model_sizes:
        model_config = config.get_model(model_size)
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: {model_size} — Data & Activations")
        logger.info(f"{'='*60}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_config.hf_id,
            device=config.device,
            dtype=config.get_dtype(),
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            cache_dir=config.cache_dir,
        )
        
        # Generate datasets (uses first model only, datasets are model-generated)
        if not args.skip_datagen and not all_datasets:
            ds_dir = os.path.join(config.output_dir, "datasets", model_size)
            all_datasets = generate_all_datasets(model, tokenizer, config, ds_dir)
        elif not all_datasets:
            # Load cached datasets
            ds_dir = os.path.join(config.output_dir, "datasets", model_size)
            for ds_name in ["repe", "role", "sand", "mask"]:
                path = os.path.join(ds_dir, f"d_{ds_name}.json")
                if os.path.exists(path):
                    all_datasets[ds_name] = DataSplit.load(path)
                    logger.info(f"Loaded cached {ds_name}: {len(all_datasets[ds_name].all_examples)} examples")
        
        # Collect activations
        if not args.skip_activation:
            collector = ActivationCollector(model, tokenizer)
            # Update config with auto-detected values
            model_config.n_layers = collector.n_layers
            model_config.d_model = collector.d_model
            
            all_activations[model_size] = {}
            
            for ds_name, ds in all_datasets.items():
                logger.info(f"Collecting activations: {model_size}/{ds_name}...")
                
                acts, labels, prompt_acts = collect_dataset_activations(
                    collector, ds.all_examples, model_size,
                    config.activation_dir, ds_name,
                )
                
                # Split into train/val/test
                all_activations[model_size][ds_name] = prepare_activation_splits(
                    acts, labels, config.data.train_frac, config.data.val_frac,
                )
        else:
            # Load cached activations
            all_activations[model_size] = {}
            for ds_name in ["repe", "role", "sand", "mask"]:
                acts_path = os.path.join(config.activation_dir, model_size, ds_name, "activations.npy")
                labels_path = os.path.join(config.activation_dir, model_size, ds_name, "labels.npy")
                if os.path.exists(acts_path):
                    acts = np.load(acts_path)
                    labels = np.load(labels_path)
                    all_activations[model_size][ds_name] = prepare_activation_splits(
                        acts, labels, config.data.train_frac, config.data.val_frac,
                    )
                    logger.info(f"Loaded cached activations: {model_size}/{ds_name} shape={acts.shape}")
        
        # Train blue team probe (for Study 3)
        if "repe" in all_activations.get(model_size, {}):
            data = all_activations[model_size]["repe"]
            train_acts, train_labels = data["train"]
            val_acts, val_labels = data["val"]
            
            if len(np.unique(train_labels)) < 2 or len(train_labels) < 10:
                logger.warning(f"Insufficient D-RepE data for {model_size}, skipping probe training")
                continue
            
            probe_results = train_probes_all_layers(
                train_acts, train_labels, config, val_acts, val_labels,
            )
            best_layer = probe_results["best_layer"]
            best_probe = probe_results["layer_probes"][best_layer]
            blue_team_probes[model_size] = (best_probe, best_layer, probe_results)
            
            logger.info(f"Blue team probe ({model_size}): layer={best_layer}, AUROC={probe_results['best_auroc']:.4f}")
        
        # Free model memory between sizes if running multiple
        if len(config.model_sizes) > 1 and model_size != config.model_sizes[-1]:
            del model
            torch.cuda.empty_cache()
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Run Studies
    # ══════════════════════════════════════════════════════════════
    
    all_results = {}
    
    # Study 1: Scaling Laws
    if config.run_study1:
        logger.info(f"\n{'#'*60}")
        logger.info("STUDY 1: Scaling Laws of Deception Representations")
        logger.info(f"{'#'*60}")
        all_results["study1"] = run_study1(all_activations, all_datasets, config)
    
    # Study 2: Geometry
    if config.run_study2:
        logger.info(f"\n{'#'*60}")
        logger.info("STUDY 2: Geometric Complexity — Subspace or Cone?")
        logger.info(f"{'#'*60}")
        all_results["study2"] = run_study2(all_activations, config)
    
    # Study 3: Adversarial Robustness (requires model in memory)
    if config.run_study3:
        logger.info(f"\n{'#'*60}")
        logger.info("STUDY 3: Adversarial Robustness & Five-Probe Battery")
        logger.info(f"{'#'*60}")
        
        # Pick the target model: prefer config setting, fall back to largest available
        rl_model_size = config.rl.target_model
        if rl_model_size not in config.model_sizes:
            rl_model_size = config.model_sizes[-1]
        logger.info(f"Study 3 target model: {rl_model_size}")
        
        model_config = config.get_model(rl_model_size)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_config.hf_id,
            device=config.device,
            dtype=config.get_dtype(),
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            cache_dir=config.cache_dir,
        )
        
        # Get or train the blue team probe
        if rl_model_size in blue_team_probes:
            probe, layer, _ = blue_team_probes[rl_model_size]
            logger.info(f"Using cached blue team probe: layer={layer}")
        else:
            # Train probe from cached activations or collect fresh
            logger.info("Training blue team probe for Study 3...")
            if rl_model_size in all_activations and "repe" in all_activations[rl_model_size]:
                data = all_activations[rl_model_size]["repe"]
            else:
                # Load cached activations
                acts_path = os.path.join(config.activation_dir, rl_model_size, "repe", "activations.npy")
                labels_path = os.path.join(config.activation_dir, rl_model_size, "repe", "labels.npy")
                if os.path.exists(acts_path):
                    acts = np.load(acts_path)
                    labels = np.load(labels_path)
                    data = prepare_activation_splits(acts, labels, config.data.train_frac, config.data.val_frac)
                    if rl_model_size not in all_activations:
                        all_activations[rl_model_size] = {}
                    all_activations[rl_model_size]["repe"] = data
                else:
                    logger.error(f"No cached activations for {rl_model_size}/repe. "
                                 f"Run without --skip-activation first.")
                    data = None
            
            if data is not None:
                train_acts, train_labels = data["train"]
                val_acts, val_labels = data["val"]
                if len(np.unique(train_labels)) >= 2 and len(train_labels) >= 10:
                    probe_results = train_probes_all_layers(
                        train_acts, train_labels, config, val_acts, val_labels,
                    )
                    probe = probe_results["layer_probes"][probe_results["best_layer"]]
                    layer = probe_results["best_layer"]
                    blue_team_probes[rl_model_size] = (probe, layer, probe_results)
                    logger.info(f"Blue team probe trained: layer={layer}, AUROC={probe_results['best_auroc']:.4f}")
                else:
                    logger.error("Insufficient data to train probe")
                    probe = None
            else:
                probe = None
        
        # Load datasets if needed
        if not all_datasets:
            ds_dir = os.path.join(config.output_dir, "datasets", rl_model_size)
            for ds_name in ["repe", "role", "sand", "mask"]:
                path = os.path.join(ds_dir, f"d_{ds_name}.json")
                if os.path.exists(path):
                    all_datasets[ds_name] = DataSplit.load(path)
        
        if probe is not None:
            # Auto-detect dimensions from model
            _tmp = ActivationCollector(model, tokenizer)
            model_config.n_layers = _tmp.n_layers
            model_config.d_model = _tmp.d_model
            del _tmp
            
            all_results["study3"] = run_study3(
                model, tokenizer, probe, layer,
                all_activations, all_datasets, config, model_config,
            )
        else:
            logger.warning("Skipping Study 3: no blue team probe available")
    
    # Study 4: Entropy
    if config.run_study4:
        logger.info(f"\n{'#'*60}")
        logger.info("STUDY 4: Entropy-Proxy Hypothesis")
        logger.info(f"{'#'*60}")
        
        # Reset CUDA state (Study 3 may have poisoned it)
        cuda_ok = True
        if torch.cuda.is_available():
            try:
                # Test if CUDA is still healthy
                _ = torch.zeros(1, device="cuda")
                del _
                torch.cuda.empty_cache()
            except (RuntimeError, Exception):
                logger.warning("CUDA context is poisoned from Study 3. Must restart process for Study 4.")
                logger.warning("Run Study 4 separately: python scripts/run_all.py --models 12B --skip-datagen --skip-activation --studies 4")
                cuda_ok = False
        
        if not cuda_ok:
            logger.error("Skipping Study 4 due to CUDA corruption. Re-run with --studies 4")
        else:
            # Need model for logit lens — always reload fresh
            target = config.model_sizes[-1]
            model_config = config.get_model(target)
            
            # Free any existing model first
            try:
                del model
                del tokenizer
            except NameError:
                pass
            torch.cuda.empty_cache()
            import gc; gc.collect()
            
            model, tokenizer = load_model_and_tokenizer(
                model_config.hf_id,
                device=config.device,
                dtype=config.get_dtype(),
                load_in_8bit=config.load_in_8bit,
                load_in_4bit=config.load_in_4bit,
                cache_dir=config.cache_dir,
            )
            all_results["study4"] = run_study4(model, all_activations, config)
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 3: SAE Analysis (if probes available)
    # ══════════════════════════════════════════════════════════════
    for model_size, (probe, layer, probe_res) in blue_team_probes.items():
        if "repe" in all_activations.get(model_size, {}):
            logger.info(f"\nSAE Analysis: {model_size}")
            data = all_activations[model_size]["repe"]
            train_acts, train_labels = data["train"]
            
            try:
                sae_results = run_sae_analysis(
                    model, tokenizer, probe,
                    train_acts, train_labels,
                    all_datasets.get("repe", DataSplit([], [], [])).train,
                    model_size, layer, config,
                )
                all_results[f"sae_{model_size}"] = sae_results
            except Exception as e:
                logger.warning(f"SAE analysis failed for {model_size}: {e}")
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Summary & Save
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    
    # Build summary
    summary = {
        "elapsed_seconds": elapsed,
        "models_tested": config.model_sizes,
        "studies_run": [i for i in [1,2,3,4] if getattr(config, f"run_study{i}")],
    }
    
    # Extract key numbers for each study
    if "study1" in all_results:
        s1 = all_results["study1"]
        summary["study1"] = {
            size: {
                "auroc": s1["scaling"][size]["test_metrics"]["auroc"]["point"],
                "best_layer": s1["scaling"][size]["best_layer"],
            }
            for size in s1["scaling"]
        }
    
    if "study2" in all_results:
        s2 = all_results["study2"]
        summary["study2"] = {
            "p1a_holds": s2.get("transfer", {}).get("p1a_holds"),
            "k_star": s2.get("pca", {}).get("k_star"),
            "geometry_conclusion": s2.get("geometry", {}).get("conclusion"),
        }
    
    if "study3" in all_results:
        s3 = all_results["study3"]
        fp = s3.get("five_probe", {})
        summary["study3"] = {
            "scenario": fp.get("scenario"),
            "scenario_desc": fp.get("scenario_desc"),
            "probe_a_auroc": fp.get("probe_a", {}).get("post_rl_auroc"),
            "probe_b_auroc": fp.get("probe_b", {}).get("post_rl_auroc"),
        }
    
    if "study4" in all_results:
        s4 = all_results["study4"]
        for ms in s4:
            if "repe" in s4[ms]:
                summary["study4"] = {
                    "model": ms,
                    "p4a_supported": s4[ms]["repe"]["correlation"]["p4a_supported"],
                    "p4b_supported": s4[ms]["repe"]["residualization"].get("p4b_supported"),
                    "interpretation": s4[ms]["repe"]["residualization"].get("interpretation"),
                }
                break
    
    all_results["summary"] = summary
    save_json(all_results, os.path.join(config.results_dir, "all_results.json"))
    save_json(summary, os.path.join(config.results_dir, "summary.json"))
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {config.results_dir}")
    logger.info(f"\nSummary:")
    for k, v in summary.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for kk, vv in v.items():
                logger.info(f"    {kk}: {vv}")
        else:
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
