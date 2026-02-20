"""
Study 2: Geometric Complexity — Subspace or Cone? (§6).
Tests H-LIN, H-SUB, H-CONE via cross-domain transfer,
multi-dim probes, PCA with permutation null, NMF, and Rayleigh test.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import KFold

from src.probes import LinearProbe, MultiDimProbe
from src.metrics import auroc, rayleigh_test, delong_test
from src.utils import save_json, set_seed

logger = logging.getLogger(__name__)


def compute_transfer_matrix(
    all_activations: Dict[str, Dict],
    model_size: str,
    config,
) -> Dict:
    """
    Compute 4x4 cross-domain transfer matrix (§6.1, Table 6).
    Tests prediction P1a: if H-LIN holds, all off-diagonal AUROC >= 0.90.
    """
    datasets = ["repe", "role", "sand", "mask"]
    matrix = {}
    probes = {}
    
    # Train best-layer probes for each dataset
    for ds_train in datasets:
        if ds_train not in all_activations.get(model_size, {}):
            continue
        
        data = all_activations[model_size][ds_train]
        train_acts, train_labels = data["train"]
        val_acts, val_labels = data["val"]
        
        # Skip datasets with fewer than 2 classes or too few samples
        if len(np.unique(train_labels)) < 2 or len(train_labels) < 10:
            logger.warning(f"Skipping {ds_train}: insufficient data (n={len(train_labels)}, classes={len(np.unique(train_labels))})")
            continue
        
        # Find best layer on this dataset (respecting min_layer to avoid prompt confound)
        n_layers = train_acts.shape[1]
        min_layer = int(n_layers * 0.3)
        best_auroc = 0
        best_layer = min_layer
        best_probe = None
        
        for layer in range(min_layer, n_layers):
            probe = LinearProbe(seed=config.probe.seeds[0])
            probe.fit(train_acts[:, layer, :], train_labels)
            val_auc = probe.score(val_acts[:, layer, :], val_labels)
            if val_auc > best_auroc:
                best_auroc = val_auc
                best_layer = layer
                best_probe = probe
        
        probes[ds_train] = {"probe": best_probe, "layer": best_layer}
    
    # Evaluate each probe on each test set
    for ds_train in datasets:
        if ds_train not in probes:
            continue
        matrix[ds_train] = {}
        probe = probes[ds_train]["probe"]
        layer = probes[ds_train]["layer"]
        
        for ds_test in datasets:
            if ds_test not in all_activations.get(model_size, {}):
                matrix[ds_train][ds_test] = None
                continue
            
            test_data = all_activations[model_size][ds_test]
            test_acts, test_labels = test_data["test"]
            
            if len(np.unique(test_labels)) < 2 or len(test_labels) < 4:
                matrix[ds_train][ds_test] = None
                continue
            
            test_scores = probe.predict_proba(test_acts[:, layer, :])
            test_auroc = auroc(test_labels, test_scores)
            matrix[ds_train][ds_test] = float(test_auroc)
    
    # Check P1a: all off-diagonal >= 0.90?
    p1a_holds = True
    for ds_train in datasets:
        for ds_test in datasets:
            if ds_train == ds_test:
                continue
            val = matrix.get(ds_train, {}).get(ds_test)
            if val is not None and val < 0.90:
                p1a_holds = False
    
    return {
        "matrix": matrix,
        "p1a_holds": p1a_holds,
        "probes": {k: {"layer": v["layer"]} for k, v in probes.items()},
    }


def pca_with_permutation_null(
    all_activations: Dict[str, Dict],
    model_size: str,
    best_layer: int,
    config,
) -> Dict:
    """
    PCA on pooled difference vectors with 1000-permutation null (§6.3, §4.9.1).
    Tests prediction P2a.
    """
    logger.info("Computing PCA with permutation null...")
    
    # Collect difference vectors from all datasets
    diff_vectors = []
    for ds_name in ["repe", "role", "sand", "mask"]:
        if ds_name not in all_activations.get(model_size, {}):
            continue
        data = all_activations[model_size][ds_name]
        # Pool train + val + test
        for split in ["train", "val", "test"]:
            acts, labels = data[split]
            if len(labels) < 4 or len(np.unique(labels)) < 2:
                continue
            X = acts[:, best_layer, :]
            # Pair honest (label=0) and deceptive (label=1)
            honest_mask = labels == 0
            deceptive_mask = labels == 1
            honest_acts = X[honest_mask]
            deceptive_acts = X[deceptive_mask]
            n_pairs = min(len(honest_acts), len(deceptive_acts))
            for j in range(n_pairs):
                diff_vectors.append(deceptive_acts[j] - honest_acts[j])
    
    diff_vectors = np.array(diff_vectors)
    n_diffs, d = diff_vectors.shape
    logger.info(f"  Collected {n_diffs} difference vectors in d={d}")
    
    # Real PCA
    max_k = min(50, n_diffs, d)
    pca = PCA(n_components=max_k)
    pca.fit(diff_vectors)
    real_eigenvalues = pca.explained_variance_
    real_var_explained = pca.explained_variance_ratio_
    
    # Permutation null (§4.9.1)
    n_perms = config.geometry.n_permutations
    null_eigenvalues = np.zeros((n_perms, max_k))
    
    rng = np.random.RandomState(42)
    for perm_i in range(n_perms):
        # Shuffle the label assignments (randomly negate some diff vectors)
        signs = rng.choice([-1, 1], size=n_diffs)
        shuffled = diff_vectors * signs[:, None]
        pca_null = PCA(n_components=max_k)
        pca_null.fit(shuffled)
        null_eigenvalues[perm_i] = pca_null.explained_variance_
        
        if (perm_i + 1) % 100 == 0:
            logger.info(f"  Permutation {perm_i + 1}/{n_perms}")
    
    # Determine k*: largest k where eigenvalue exceeds 95th percentile of null
    null_95th = np.percentile(null_eigenvalues, 95, axis=0)
    k_star = 0
    for k in range(max_k):
        if real_eigenvalues[k] > null_95th[k]:
            k_star = k + 1
        else:
            break
    
    logger.info(f"  k* = {k_star} significant components")
    
    return {
        "n_diff_vectors": n_diffs,
        "d": d,
        "k_star": k_star,
        "real_eigenvalues": real_eigenvalues.tolist(),
        "real_var_explained": real_var_explained.tolist(),
        "null_95th_percentile": null_95th.tolist(),
        "diff_vectors": diff_vectors,  # keep for further analysis
    }


def cone_vs_subspace_test(
    diff_vectors: np.ndarray,
    k_star: int,
    config,
) -> Dict:
    """
    Rayleigh test + NMF vs PCA comparison (§6.4, §4.9.2).
    Tests predictions P3a (directional asymmetry) and P3b (NMF >= PCA).
    """
    logger.info("Running Cone vs. Subspace tests...")
    results = {}
    
    if k_star < 1:
        logger.warning("k_star < 1, cannot run geometric tests")
        return {"error": "k_star < 1"}
    
    # Project into k* subspace
    pca = PCA(n_components=k_star)
    projected = pca.fit_transform(diff_vectors)  # (n, k_star)
    
    # ── Rayleigh test for directional asymmetry (P3a) ──
    # Normalize projected vectors to unit sphere
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_dirs = projected / norms
    
    R_bar, p_value = rayleigh_test(unit_dirs)
    p3a_holds = p_value < config.geometry.rayleigh_alpha
    
    results["rayleigh"] = {
        "R_bar": float(R_bar),
        "p_value": float(p_value),
        "alpha": config.geometry.rayleigh_alpha,
        "p3a_supported": p3a_holds,
    }
    logger.info(f"  Rayleigh: R_bar={R_bar:.4f}, p={p_value:.2e}, P3a={'YES' if p3a_holds else 'NO'}")
    
    # ── NMF vs PCA cross-validated reconstruction (P3b) ──
    # NMF requires non-negative data, so we shift diff_vectors
    # This is the key test: if deception directions form a cone (non-negative combinations),
    # NMF should fit as well as PCA
    
    # Approach: Compare on the POSITIVE part of the projected data
    # Shift to make all values positive for NMF
    shift = np.abs(projected.min()) + 1e-6
    projected_pos = projected + shift
    
    kf = KFold(n_splits=config.geometry.nmf_cv_folds, shuffle=True, random_state=42)
    pca_r2s = []
    nmf_r2s = []
    
    for train_idx, test_idx in kf.split(projected_pos):
        X_train = projected_pos[train_idx]
        X_test = projected_pos[test_idx]
        
        # PCA reconstruction
        pca_inner = PCA(n_components=k_star)
        pca_inner.fit(X_train)
        X_pca_recon = pca_inner.inverse_transform(pca_inner.transform(X_test))
        ss_res_pca = np.sum((X_test - X_pca_recon) ** 2)
        ss_tot = np.sum((X_test - X_test.mean(axis=0)) ** 2)
        pca_r2 = 1 - ss_res_pca / ss_tot if ss_tot > 0 else 0
        pca_r2s.append(pca_r2)
        
        # NMF reconstruction
        try:
            nmf = NMF(n_components=k_star, max_iter=500, random_state=42)
            W_train = nmf.fit_transform(X_train)
            H = nmf.components_
            W_test = nmf.transform(X_test)
            X_nmf_recon = W_test @ H
            ss_res_nmf = np.sum((X_test - X_nmf_recon) ** 2)
            nmf_r2 = 1 - ss_res_nmf / ss_tot if ss_tot > 0 else 0
            nmf_r2s.append(nmf_r2)
        except Exception as e:
            logger.warning(f"NMF failed: {e}")
            nmf_r2s.append(0.0)
    
    pca_mean_r2 = np.mean(pca_r2s)
    nmf_mean_r2 = np.mean(nmf_r2s)
    p3b_holds = nmf_mean_r2 >= pca_mean_r2 - 0.05  # NMF comparable or better
    
    results["nmf_vs_pca"] = {
        "pca_r2_mean": float(pca_mean_r2),
        "pca_r2_std": float(np.std(pca_r2s)),
        "nmf_r2_mean": float(nmf_mean_r2),
        "nmf_r2_std": float(np.std(nmf_r2s)),
        "p3b_supported": p3b_holds,
    }
    logger.info(f"  PCA R²={pca_mean_r2:.4f}, NMF R²={nmf_mean_r2:.4f}, P3b={'YES' if p3b_holds else 'NO'}")
    
    # ── Overall geometric conclusion ──
    if p3a_holds and p3b_holds:
        conclusion = "H-CONE"
    elif not p3a_holds:
        conclusion = "H-SUB"
    else:
        conclusion = "AMBIGUOUS"
    
    results["conclusion"] = conclusion
    logger.info(f"  Geometric conclusion: {conclusion}")
    
    return results


def run_study2(
    all_activations: Dict[str, Dict],
    config,
) -> Dict:
    """Run all Study 2 analyses."""
    results = {}
    save_dir = Path(config.results_dir) / "study2"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    target_model = "27B" if "27B" in config.model_sizes else config.model_sizes[-1]
    logger.info(f"\nStudy 2: Geometric analysis on {target_model}")
    
    # 6.1: Transfer matrix
    logger.info("Computing transfer matrix...")
    results["transfer"] = compute_transfer_matrix(all_activations, target_model, config)
    
    # Get best layer from repe probe
    repe_data = all_activations.get(target_model, {}).get("repe", {})
    if "train" in repe_data:
        train_acts, train_labels = repe_data["train"]
        val_acts, val_labels = repe_data["val"]
        
        # Quick best-layer search (respecting min_layer)
        n_layers = train_acts.shape[1]
        min_layer = int(n_layers * 0.3)
        best_layer = min_layer
        best_auc = 0
        for l in range(min_layer, n_layers):
            probe = LinearProbe(seed=42)
            probe.fit(train_acts[:, l, :], train_labels)
            auc = probe.score(val_acts[:, l, :], val_labels)
            if auc > best_auc:
                best_auc = auc
                best_layer = l
    else:
        best_layer = 0
    
    # 6.3: PCA with permutation null
    pca_results = pca_with_permutation_null(all_activations, target_model, best_layer, config)
    results["pca"] = {k: v for k, v in pca_results.items() if k != "diff_vectors"}
    
    # 6.4: Cone vs. Subspace
    if pca_results["k_star"] > 0:
        results["geometry"] = cone_vs_subspace_test(
            pca_results["diff_vectors"],
            pca_results["k_star"],
            config,
        )
    
    save_json(
        {k: v for k, v in results.items()},
        str(save_dir / "study2_results.json"),
    )
    logger.info(f"Study 2 results saved to {save_dir}")
    
    return results
