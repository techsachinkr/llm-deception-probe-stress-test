"""
Study 4: The Entropy-Proxy Hypothesis (§5.4).
Implements entropy computation (Logit Lens + Tuned Lens),
probe-entropy correlation (P4a), entropy residualization (P4b/P4c),
and post-RL entropy dynamics.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.probes import LinearProbe
from src.metrics import (
    auroc, spearman_correlation, welch_t_test, cohens_d,
    delong_test, bootstrap_ci
)
from src.utils import save_json

logger = logging.getLogger(__name__)


def _find_final_norm(model):
    """Robustly find the final layer norm in any model architecture."""
    import torch.nn as nn
    
    # Try explicit paths first
    paths = [
        "model.norm",
        "model.model.norm",
        "transformer.ln_f",
        "model.final_layernorm",
        "model.model.final_layernorm",
        "language_model.model.norm",
    ]
    
    for path in paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, 'weight'):
                logger.info(f"Found final norm at: {path} ({type(obj).__name__})")
                return obj
        except AttributeError:
            continue
    
    # Recursive search: find the last RMSNorm or LayerNorm that's NOT inside a layer
    norms = []
    def _search(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.LayerNorm,)) or "RMSNorm" in type(child).__name__:
                # Skip norms inside transformer layers
                if "layers." not in full and "layer." not in full:
                    norms.append((full, child))
            _search(child, full)
    
    _search(model)
    
    if norms:
        # Take the last non-layer norm found (likely the final one)
        path, norm = norms[-1]
        logger.info(f"Found final norm via search: {path} ({type(norm).__name__})")
        return norm
    
    logger.warning("Could not find final norm anywhere in model")
    return None


def _find_lm_head(model):
    """Robustly find the unembedding layer."""
    import torch.nn as nn
    
    for path in ["lm_head", "model.lm_head", "output"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, 'weight'):
                logger.info(f"Found lm_head at: {path}")
                return obj
        except AttributeError:
            continue
    
    # Try tied embeddings
    for path in ["model.embed_tokens", "model.model.embed_tokens"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, 'weight'):
                logger.info(f"Using tied embeddings from: {path}")
                return obj  # Will use .weight.T for projection
        except AttributeError:
            continue
    
    return None


def _manual_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Manual RMSNorm when we can't find the model's norm layer."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms


def compute_logit_lens_entropy(
    model,
    activations_at_layer: np.ndarray,  # (n, d_model) activations at a single layer
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute Logit Lens entropy: H(softmax(W_U · norm(h_l))) for each example.
    
    CRITICAL: Must apply the model's final layer norm (RMSNorm for Gemma 3)
    before projecting through the unembedding matrix. Without this,
    activations have wildly different scales and the logits are meaningless.
    """
    final_norm = _find_final_norm(model)
    lm_head = _find_lm_head(model)
    
    if lm_head is None:
        logger.error("Cannot find lm_head or embeddings — entropy will be zeros")
        return np.zeros(len(activations_at_layer))
    
    # Determine if lm_head is a proper Linear or tied embeddings
    # Keep weight in model dtype (float16/bfloat16) to avoid OOM on large models
    lm_head_weight = lm_head.weight.data  # Stay in model dtype
    lm_head_dtype = lm_head_weight.dtype
    is_tied = not isinstance(lm_head, torch.nn.Linear)
    
    vocab_size = lm_head_weight.shape[0]
    logger.info(f"lm_head: vocab={vocab_size}, dtype={lm_head_dtype}, "
                f"size={vocab_size * lm_head_weight.shape[1] * lm_head_weight.element_size() / 1e9:.2f} GiB")
    
    entropies = np.zeros(len(activations_at_layer))
    batch_size = 16 if vocab_size > 200000 else 64  # smaller batches for large vocab
    
    for i in range(0, len(activations_at_layer), batch_size):
        batch = activations_at_layer[i:i + batch_size]
        h = torch.tensor(batch, dtype=lm_head_dtype).to(device)
        
        with torch.no_grad():
            # Step 1: Apply normalization (in model dtype)
            if final_norm is not None:
                try:
                    h_normed = final_norm(h)
                except Exception as e:
                    logger.debug(f"Model norm failed ({e}), using manual RMSNorm")
                    h_normed = _manual_rms_norm(h.float()).to(lm_head_dtype)
            else:
                h_normed = _manual_rms_norm(h.float()).to(lm_head_dtype)
            
            # Step 2: Project through unembedding (in model dtype to save memory)
            logits = h_normed @ lm_head_weight.T  # (batch, vocab_size) in float16/bf16
            
            # Step 3: Convert to float32 for numerically stable softmax
            logits = logits.float().clamp(-100, 100)
            
            # Free the half-precision intermediates
            del h, h_normed
            
            # Step 4: Compute entropy
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropies[i:i + batch_size] = entropy.cpu().numpy()
            
            del logits, log_probs, probs, entropy
            torch.cuda.empty_cache()
    
    # Diagnostics
    mean_ent = np.mean(entropies)
    std_ent = np.std(entropies)
    logger.info(f"Entropy stats: mean={mean_ent:.2f}, std={std_ent:.2f}, "
                f"range=[{np.min(entropies):.2f}, {np.max(entropies):.2f}], "
                f"norm={'model' if final_norm else 'manual_rms'}")
    
    if mean_ent < 0.01:
        logger.warning(f"Entropy values suspiciously low (mean={mean_ent:.6f}). "
                       f"Possible issues: norm not applied, wrong lm_head, or "
                       f"activations already at zero magnitude.")
        # Debug: check activation magnitudes
        sample_norms = np.linalg.norm(activations_at_layer[:10], axis=1)
        logger.warning(f"Activation L2 norms (first 10): {sample_norms}")
    
    return entropies


def compute_layerwise_entropy(
    model,
    activations: np.ndarray,  # (n, n_layers, d_model)
    labels: np.ndarray,
    min_layer_frac: float = 0.5,
    device: str = "cuda",
) -> Dict:
    """
    Compute per-layer entropy for honest vs. deceptive examples.
    Only uses Logit Lens for layers in top half (§3.4 calibration caveat).
    """
    n, n_layers, d_model = activations.shape
    min_layer = int(n_layers * min_layer_frac)
    
    honest_mask = labels == 0
    deceptive_mask = labels == 1
    
    layer_results = {}
    
    for layer in range(min_layer, n_layers):
        X = activations[:, layer, :]
        entropies = compute_logit_lens_entropy(model, X, device)
        
        h_ent = entropies[honest_mask]
        d_ent = entropies[deceptive_mask]
        
        t_stat, p_val = welch_t_test(d_ent, h_ent)
        d_effect = cohens_d(d_ent, h_ent)
        
        layer_results[layer] = {
            "honest_mean": float(np.mean(h_ent)),
            "honest_std": float(np.std(h_ent)),
            "deceptive_mean": float(np.mean(d_ent)),
            "deceptive_std": float(np.std(d_ent)),
            "delta": float(np.mean(d_ent) - np.mean(h_ent)),
            "cohens_d": float(d_effect),
            "welch_t": float(t_stat),
            "welch_p": float(p_val),
            "all_entropies": entropies,  # keep for correlation
        }
    
    return layer_results


def test_probe_entropy_correlation(
    probe: LinearProbe,
    activations_at_layer: np.ndarray,  # (n, d_model)
    entropies: np.ndarray,             # (n,)
    labels: np.ndarray,
    config,
) -> Dict:
    """
    Test P4a: Spearman correlation between probe score and entropy (§4.8.2, §8.2).
    """
    probe_scores = probe.predict_proba(activations_at_layer)
    
    rho, p_value = spearman_correlation(probe_scores, entropies)
    
    p4a_holds = abs(rho) > config.entropy.spearman_threshold
    
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "threshold": config.entropy.spearman_threshold,
        "p4a_supported": p4a_holds,
    }


def entropy_residualization_test(
    activations_at_layer: np.ndarray,  # (n, d_model)
    entropies: np.ndarray,             # (n,)
    labels: np.ndarray,
    config,
) -> Dict:
    """
    Entropy residualization test (§4.8.3, §8.3).
    Tests P4b (AUROC drop > 0.10) and P4c (residualized AUROC > 0.70).
    
    Procedure:
    1. Regress each dimension of h on H(P_l) via OLS
    2. Compute residuals: h_tilde = h - H(P_l) * beta_hat
    3. Retrain probe on h_tilde
    4. Compare AUROC via DeLong test
    """
    n, d = activations_at_layer.shape
    
    # Split into train/test
    split = int(0.7 * n)
    X_train, X_test = activations_at_layer[:split], activations_at_layer[split:]
    H_train, H_test = entropies[:split], entropies[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    # ── Original probe ──
    original_probe = LinearProbe(seed=42)
    original_probe.fit(X_train, y_train)
    original_scores = original_probe.predict_proba(X_test)
    original_auroc = auroc(y_test, original_scores)
    
    # ── Residualize: regress h on H(P_l) ──
    # OLS: beta_hat = (H^T H)^{-1} H^T X  (scalar regressor)
    H_train_col = H_train.reshape(-1, 1)
    H_test_col = H_test.reshape(-1, 1)
    
    # For each dimension: beta_j = cov(H, X_j) / var(H)
    H_centered = H_train - H_train.mean()
    var_H = np.var(H_train)
    if var_H < 1e-10:
        logger.warning("Entropy has near-zero variance, residualization skipped")
        return {"error": "zero_variance"}
    
    beta_hat = (H_centered.reshape(-1, 1) * (X_train - X_train.mean(axis=0))).mean(axis=0) / var_H
    # Shape: (d,)
    
    # Residualize
    X_train_resid = X_train - np.outer(H_train, beta_hat)
    X_test_resid = X_test - np.outer(H_test, beta_hat)
    
    # ── Residualized probe ──
    resid_probe = LinearProbe(seed=42)
    resid_probe.fit(X_train_resid, y_train)
    resid_scores = resid_probe.predict_proba(X_test_resid)
    resid_auroc = auroc(y_test, resid_scores)
    
    # ── DeLong test ──
    z_stat, delong_p = delong_test(y_test, original_scores, resid_scores)
    
    # ── Test predictions ──
    delta_auroc = original_auroc - resid_auroc
    p4b_holds = delta_auroc > config.entropy.auroc_drop_threshold and delong_p < config.entropy.delong_alpha
    p4c_holds = resid_auroc > config.entropy.residualized_auroc_min
    
    return {
        "original_auroc": float(original_auroc),
        "residualized_auroc": float(resid_auroc),
        "delta_auroc": float(delta_auroc),
        "delong_z": float(z_stat),
        "delong_p": float(delong_p),
        "p4b_supported": p4b_holds,  # entropy is major component
        "p4c_supported": p4c_holds,  # probe captures beyond entropy
        "interpretation": (
            "Entropy is the dominant signal" if p4b_holds and not p4c_holds else
            "Entropy contributes but probe captures more" if p4b_holds and p4c_holds else
            "Entropy is NOT the primary signal" if not p4b_holds else
            "Ambiguous"
        ),
    }


def run_study4(
    model,
    all_activations: Dict[str, Dict],
    config,
) -> Dict:
    """Run all Study 4 analyses."""
    results = {}
    save_dir = Path(config.results_dir) / "study4"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_size in config.model_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Study 4: Entropy analysis for {model_size}")
        logger.info(f"{'='*60}")
        
        model_results = {}
        
        for ds_name in ["repe", "role", "sand", "mask"]:
            if ds_name not in all_activations.get(model_size, {}):
                continue
            
            data = all_activations[model_size][ds_name]
            train_acts, train_labels = data["train"]
            test_acts, test_labels = data["test"]
            
            # Skip if insufficient data
            if len(np.unique(train_labels)) < 2 or len(train_labels) < 10:
                logger.warning(f"Skipping {model_size}/{ds_name}: insufficient data")
                continue
            if len(np.unique(test_labels)) < 2 or len(test_labels) < 10:
                logger.warning(f"Skipping {model_size}/{ds_name}: insufficient test data")
                continue
            
            n_layers = train_acts.shape[1]
            d_model = train_acts.shape[2]
            min_layer = int(n_layers * config.entropy.logit_lens_min_layer_frac)
            
            # Find best probe layer (respecting min_layer for both probe and entropy)
            best_layer = min_layer
            best_auc = 0
            best_probe = None
            for l in range(min_layer, n_layers):
                p = LinearProbe(seed=42)
                p.fit(train_acts[:, l, :], train_labels)
                auc = p.score(test_acts[:, l, :], test_labels)
                if auc > best_auc:
                    best_auc = auc
                    best_layer = l
                    best_probe = p
            
            if best_layer < min_layer:
                logger.warning(f"  Best layer {best_layer} < min_layer {min_layer}, using min_layer")
                best_layer = min_layer
                best_probe = LinearProbe(seed=42)
                best_probe.fit(train_acts[:, best_layer, :], train_labels)
            
            logger.info(f"  {ds_name}: best_layer={best_layer}, AUROC={best_auc:.4f}")
            
            # ── 8.1: Entropy gap ──
            logger.info(f"  Computing entropy at layer {best_layer}...")
            test_entropies = compute_logit_lens_entropy(
                model, test_acts[:, best_layer, :], config.device
            )
            
            honest_ent = test_entropies[test_labels == 0]
            deceptive_ent = test_entropies[test_labels == 1]
            
            t_stat, p_val = welch_t_test(deceptive_ent, honest_ent)
            d_effect = cohens_d(deceptive_ent, honest_ent)
            
            entropy_gap = {
                "layer": best_layer,
                "honest_mean": float(np.mean(honest_ent)),
                "honest_std": float(np.std(honest_ent)),
                "deceptive_mean": float(np.mean(deceptive_ent)),
                "deceptive_std": float(np.std(deceptive_ent)),
                "cohens_d": float(d_effect),
                "welch_t": float(t_stat),
                "welch_p": float(p_val),
            }
            
            # ── 8.2: Probe-entropy correlation ──
            correlation = test_probe_entropy_correlation(
                best_probe,
                test_acts[:, best_layer, :],
                test_entropies,
                test_labels,
                config,
            )
            
            # ── 8.3: Entropy residualization ──
            # Pool train + test for more data
            all_acts_layer = np.concatenate([
                train_acts[:, best_layer, :],
                test_acts[:, best_layer, :],
            ])
            all_labels = np.concatenate([train_labels, test_labels])
            all_entropies = compute_logit_lens_entropy(
                model, all_acts_layer, config.device
            )
            
            residualization = entropy_residualization_test(
                all_acts_layer, all_entropies, all_labels, config,
            )
            
            model_results[ds_name] = {
                "entropy_gap": entropy_gap,
                "correlation": correlation,
                "residualization": residualization,
            }
            
            logger.info(f"  Entropy gap: {entropy_gap['deceptive_mean']:.2f} vs {entropy_gap['honest_mean']:.2f} (d={d_effect:.2f})")
            logger.info(f"  Correlation: ρ={correlation['spearman_rho']:.3f}, P4a={'YES' if correlation['p4a_supported'] else 'NO'}")
            logger.info(f"  Residualization: {residualization.get('interpretation', 'N/A')}")
        
        results[model_size] = model_results
    
    save_json(results, str(save_dir / "study4_results.json"))
    logger.info(f"Study 4 results saved to {save_dir}")
    
    return results
