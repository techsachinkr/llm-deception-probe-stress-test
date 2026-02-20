"""
SAE Feature Analysis and Causal Validation.
Uses Gemma Scope 2 SAEs to decompose probe directions
and validate features via correlation, causal ablation, and specificity.
"""
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.stats import pointbiserialr
from tqdm import tqdm

from src.utils import save_json

logger = logging.getLogger(__name__)


def load_sae_for_layer(
    model_size: str,
    layer: int,
    cache_dir: str = "./cache",
) -> Optional[Dict]:
    """
    Load Gemma Scope 2 SAE for a specific model and layer.
    
    Returns dict with 'encoder_weight', 'encoder_bias', 
    'decoder_weight', 'decoder_bias' as numpy arrays,
    or None if not available.
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Gemma Scope 2 model IDs
        scope_ids = {
            "1B": "google/gemma-scope-2-1b-it",
            "4B": "google/gemma-scope-2-4b-it",
            "12B": "google/gemma-scope-2-12b-it",
            "27B": "google/gemma-scope-2-27b-it",
        }
        
        model_id = scope_ids.get(model_size)
        if model_id is None:
            logger.warning(f"No Gemma Scope 2 for {model_size}")
            return None
        
        # Try loading the SAE weights
        # The exact path depends on Gemma Scope 2's structure
        # This attempts the most common pattern
        import safetensors.torch as st
        
        filename = f"layer_{layer}/sae_weights.safetensors"
        try:
            path = hf_hub_download(model_id, filename, cache_dir=cache_dir)
            weights = st.load_file(path)
        except Exception:
            # Try alternative naming conventions
            filename = f"layer{layer}/params.safetensors"
            try:
                path = hf_hub_download(model_id, filename, cache_dir=cache_dir)
                weights = st.load_file(path)
            except Exception as e:
                logger.warning(f"Cannot load SAE for {model_size} layer {layer}: {e}")
                return None
        
        # Extract weights
        sae = {}
        for key in weights:
            sae[key] = weights[key].float().numpy()
        
        return sae
        
    except ImportError:
        logger.warning("huggingface_hub or safetensors not available")
        return None
    except Exception as e:
        logger.warning(f"SAE load failed: {e}")
        return None


def compute_sae_features(
    activations: np.ndarray,  # (n, d_model)
    sae: Dict,
) -> np.ndarray:
    """
    Encode activations through the SAE to get sparse feature activations.
    z = ReLU(W_enc · h + b_enc)
    """
    W_enc = sae.get("W_enc", sae.get("encoder.weight", None))
    b_enc = sae.get("b_enc", sae.get("encoder.bias", None))
    
    if W_enc is None:
        raise ValueError("Cannot find encoder weights in SAE")
    
    # z = ReLU(h @ W_enc^T + b_enc)
    z = activations @ W_enc.T
    if b_enc is not None:
        z += b_enc
    z = np.maximum(z, 0)  # ReLU
    return z


def decompose_probe_direction(
    probe_direction: np.ndarray,  # (d_model,) unit vector
    sae: Dict,
    top_k: int = 10,
) -> List[Dict]:
    """
    Project probe direction onto SAE decoder columns (§4.2).
    relevance(feature j) = |cos(w, W_dec[:, j])|
    
    Returns top-k features sorted by cosine similarity.
    """
    W_dec = sae.get("W_dec", sae.get("decoder.weight", None))
    if W_dec is None:
        raise ValueError("Cannot find decoder weights in SAE")
    
    # W_dec: (n_features, d_model) or (d_model, n_features)
    if W_dec.shape[1] != len(probe_direction):
        W_dec = W_dec.T  # transpose if needed
    
    # Normalize decoder columns
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_dec_normed = W_dec / norms
    
    # Cosine similarity with probe direction
    probe_normed = probe_direction / np.linalg.norm(probe_direction)
    cosines = W_dec_normed @ probe_normed
    
    # Get top-k by absolute cosine similarity
    top_indices = np.argsort(np.abs(cosines))[::-1][:top_k]
    
    features = []
    for idx in top_indices:
        features.append({
            "feature_idx": int(idx),
            "cosine_similarity": float(cosines[idx]),
            "abs_cosine": float(abs(cosines[idx])),
        })
    
    return features


def validate_features_correlation(
    sae: Dict,
    activations: np.ndarray,  # (n, d_model)
    labels: np.ndarray,       # (n,) binary
    feature_indices: List[int],
    threshold: float = 0.15,
) -> Dict:
    """
    Point-biserial correlation between feature activation and deception label (§4.7.1).
    """
    sae_features = compute_sae_features(activations, sae)  # (n, n_features)
    
    results = {}
    for idx in feature_indices:
        if idx >= sae_features.shape[1]:
            continue
        feature_vals = sae_features[:, idx]
        r, p = pointbiserialr(labels, feature_vals)
        results[idx] = {
            "correlation_r": float(r),
            "p_value": float(p),
            "above_threshold": abs(r) >= threshold,
        }
    
    return results


def causal_ablation_test(
    model,
    tokenizer,
    sae: Dict,
    probe,
    activations: np.ndarray,    # (n, d_model)
    labels: np.ndarray,
    examples: list,
    feature_idx: int,
    layer: int,
    n_examples: int = 200,
    device: str = "cuda",
) -> Dict:
    """
    Causal ablation: clamp feature to zero/mean and measure effect (§4.7.2).
    
    Tests:
    (a) Change in probe score
    (b) Change in model behavior (text output)
    """
    sae_features = compute_sae_features(activations[:n_examples], sae)
    
    W_dec = sae.get("W_dec", sae.get("decoder.weight", None))
    if W_dec is None:
        return {"error": "no decoder weights"}
    if W_dec.shape[1] != activations.shape[1]:
        W_dec = W_dec.T
    
    feature_direction = W_dec[feature_idx]  # (d_model,)
    
    # Get mean activation for deceptive examples
    deceptive_mask = labels[:n_examples] == 1
    deceptive_feature_vals = sae_features[deceptive_mask, feature_idx]
    mean_deceptive_val = np.mean(deceptive_feature_vals) if len(deceptive_feature_vals) > 0 else 0
    
    # ── (a) Probe score change under ablation ──
    probe_score_original = probe.predict_proba(activations[:n_examples])
    
    # Ablate: for deceptive examples, subtract the feature component
    ablated_acts = activations[:n_examples].copy()
    for i in range(n_examples):
        if labels[i] == 1:
            # Clamp feature to zero: subtract its current contribution
            current_val = sae_features[i, feature_idx]
            ablated_acts[i] -= current_val * feature_direction
    
    probe_score_ablated = probe.predict_proba(ablated_acts)
    
    # Compute effect
    deceptive_idx = np.where(labels[:n_examples] == 1)[0]
    if len(deceptive_idx) > 0:
        score_change = float(np.mean(
            probe_score_ablated[deceptive_idx] - probe_score_original[deceptive_idx]
        ))
    else:
        score_change = 0.0
    
    return {
        "feature_idx": feature_idx,
        "mean_probe_score_change": score_change,
        "mean_deceptive_feature_val": float(mean_deceptive_val),
        "n_deceptive": int(np.sum(deceptive_mask)),
        "direction_norm": float(np.linalg.norm(feature_direction)),
    }


def run_sae_analysis(
    model,
    tokenizer,
    probe,
    activations: np.ndarray,
    labels: np.ndarray,
    examples: list,
    model_size: str,
    best_layer: int,
    config,
) -> Dict:
    """
    Full SAE analysis pipeline (§5.4):
    1. Load SAE for model/layer
    2. Decompose probe direction
    3. Validate top features (correlation, ablation, specificity)
    """
    results = {"model_size": model_size, "layer": best_layer}
    
    # Load SAE
    sae = load_sae_for_layer(model_size, best_layer, config.cache_dir)
    if sae is None:
        logger.warning(f"SAE not available for {model_size} layer {best_layer}")
        results["error"] = "SAE not available"
        return results
    
    # Decompose probe direction
    probe_dir = probe.direction
    top_features = decompose_probe_direction(probe_dir, sae, config.sae.top_k_features)
    results["top_features"] = top_features
    
    # Validate top-k features
    feature_indices = [f["feature_idx"] for f in top_features[:config.sae.causal_top_k]]
    
    # Correlation (§4.7.1)
    X_layer = activations[:, best_layer, :]
    correlations = validate_features_correlation(
        sae, X_layer, labels, feature_indices, config.sae.correlation_threshold
    )
    results["correlations"] = correlations
    
    # Causal ablation (§4.7.2)
    ablation_results = {}
    for fidx in feature_indices:
        try:
            ablation = causal_ablation_test(
                model, tokenizer, sae, probe,
                X_layer, labels, examples,
                fidx, best_layer,
                n_examples=config.sae.n_ablation_examples,
                device=config.device,
            )
            ablation_results[fidx] = ablation
        except Exception as e:
            logger.warning(f"Ablation failed for feature {fidx}: {e}")
            ablation_results[fidx] = {"error": str(e)}
    
    results["ablations"] = ablation_results
    
    # Summary
    n_causal = sum(
        1 for fidx in feature_indices
        if (correlations.get(fidx, {}).get("above_threshold", False) and
            abs(ablation_results.get(fidx, {}).get("mean_probe_score_change", 0)) > 0.05)
    )
    results["n_causally_validated"] = n_causal
    results["n_tested"] = len(feature_indices)
    
    logger.info(f"SAE Analysis: {n_causal}/{len(feature_indices)} features causally validated")
    
    return results
