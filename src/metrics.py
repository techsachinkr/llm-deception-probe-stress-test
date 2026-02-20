"""
Statistical metrics: AUROC, bootstrap CIs, DeLong test, effect sizes.
Implements all metrics from §4.5.4.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score

logger = logging.getLogger(__name__)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC, handling edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def accuracy_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Compute accuracy at a specific FPR threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find the threshold closest to target_fpr
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1
    thresh = thresholds[idx]
    preds = (y_score >= thresh).astype(int)
    return accuracy_score(y_true, preds)


def f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute F1 at optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (y_score >= thresh).astype(int)
        f = f1_score(y_true, preds, zero_division=0.0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    return best_f1, best_thresh


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    
    boot_values = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            boot_values[i] = metric_fn(y_true[idx], y_score[idx])
        except (ValueError, ZeroDivisionError):
            boot_values[i] = np.nan
    
    boot_values = boot_values[~np.isnan(boot_values)]
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_values, 100 * alpha / 2)
    ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
    
    return point, ci_lower, ci_upper


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUROCs on the same data.
    Returns (z_statistic, p_value).
    
    Simplified implementation using the Mann-Whitney approach.
    """
    # Separate positive and negative scores
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    scores_a_pos = y_score_a[pos_mask]
    scores_a_neg = y_score_a[neg_mask]
    scores_b_pos = y_score_b[pos_mask]
    scores_b_neg = y_score_b[neg_mask]
    
    n_pos = len(scores_a_pos)
    n_neg = len(scores_a_neg)
    
    # Compute placement values (structural components)
    def compute_placements(pos_scores, neg_scores):
        V_pos = np.zeros(n_pos)
        V_neg = np.zeros(n_neg)
        for i in range(n_pos):
            V_pos[i] = np.mean(pos_scores[i] > neg_scores) + 0.5 * np.mean(pos_scores[i] == neg_scores)
        for j in range(n_neg):
            V_neg[j] = np.mean(pos_scores > neg_scores[j]) + 0.5 * np.mean(pos_scores == neg_scores[j])
        return V_pos, V_neg
    
    Va_pos, Va_neg = compute_placements(scores_a_pos, scores_a_neg)
    Vb_pos, Vb_neg = compute_placements(scores_b_pos, scores_b_neg)
    
    # AUC estimates
    auc_a = np.mean(Va_pos)
    auc_b = np.mean(Vb_pos)
    
    # Covariance matrix of (AUC_a, AUC_b) 
    S_pos = np.cov(np.stack([Va_pos, Vb_pos]))
    S_neg = np.cov(np.stack([Va_neg, Vb_neg]))
    
    S = S_pos / n_pos + S_neg / n_neg
    
    # Test statistic
    diff = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    
    if var_diff <= 0:
        return 0.0, 1.0
    
    z = diff / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))
    
    return z, p_value


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 10000,
    text_baseline_auroc: Optional[float] = None,
) -> Dict:
    """
    Compute all metrics from §4.5.4:
    AUROC, Acc@1%FPR, Acc@5%FPR, F1, Black-to-White Boost.
    All with bootstrap 95% CIs.
    """
    results = {}
    
    # AUROC with CI
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, auroc, n_bootstrap)
    results["auroc"] = {"point": auc_point, "ci_lower": auc_lo, "ci_upper": auc_hi}
    
    # Accuracy at 1% FPR
    acc1_fn = lambda yt, ys: accuracy_at_fpr(yt, ys, 0.01)
    acc1_point, acc1_lo, acc1_hi = bootstrap_ci(y_true, y_score, acc1_fn, n_bootstrap)
    results["acc_at_1pct_fpr"] = {"point": acc1_point, "ci_lower": acc1_lo, "ci_upper": acc1_hi}
    
    # Accuracy at 5% FPR
    acc5_fn = lambda yt, ys: accuracy_at_fpr(yt, ys, 0.05)
    acc5_point, acc5_lo, acc5_hi = bootstrap_ci(y_true, y_score, acc5_fn, n_bootstrap)
    results["acc_at_5pct_fpr"] = {"point": acc5_point, "ci_lower": acc5_lo, "ci_upper": acc5_hi}
    
    # F1 at optimal threshold
    f1_val, threshold = f1_optimal(y_true, y_score)
    results["f1_optimal"] = {"point": f1_val, "threshold": threshold}
    
    # Black-to-White Boost
    if text_baseline_auroc is not None:
        boost = auc_point - text_baseline_auroc
        results["b2w_boost"] = {"point": boost, "baseline": text_baseline_auroc}
    
    return results


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rank correlation with p-value."""
    rho, p = stats.spearmanr(x, y)
    return rho, p


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def welch_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test (unequal variances)."""
    t, p = stats.ttest_ind(group1, group2, equal_var=False)
    return t, p


def rayleigh_test(directions: np.ndarray) -> Tuple[float, float]:
    """
    Rayleigh test for directional non-uniformity on the unit sphere (§4.9.2).
    Tests H0: directions are uniformly distributed on the sphere.
    
    Args:
        directions: (n, d) array of unit vectors
    Returns:
        (R_bar, p_value)
    """
    n, d = directions.shape
    # Compute mean resultant length
    mean_direction = np.mean(directions, axis=0)
    R_bar = np.linalg.norm(mean_direction)
    
    # For high dimensions, use the chi-squared approximation
    # Test statistic: n * d * R_bar^2 ~ chi^2(d)
    test_stat = n * d * R_bar ** 2
    p_value = 1 - stats.chi2.cdf(test_stat, df=d)
    
    return R_bar, p_value


def bootstrap_peak_layer(
    layer_aurocs: np.ndarray,
    y_true: np.ndarray,
    layer_scores: Dict[int, np.ndarray],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[int, float, float]:
    """
    Bootstrap the peak layer with uncertainty (§4.9.3).
    
    Args:
        layer_aurocs: (n_layers,) array of per-layer AUROC
        y_true: (n_examples,) true labels
        layer_scores: dict mapping layer_idx -> (n_examples,) probe scores
        
    Returns:
        (median_peak_layer, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_layers = len(layer_aurocs)
    
    peak_layers = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_aurocs = []
        for l in range(n_layers):
            if l in layer_scores:
                try:
                    boot_aurocs.append(auroc(y_true[idx], layer_scores[l][idx]))
                except (ValueError, IndexError):
                    boot_aurocs.append(0.5)
            else:
                boot_aurocs.append(0.5)
        peak_layers.append(np.argmax(boot_aurocs))
    
    peak_layers = np.array(peak_layers)
    median_peak = np.median(peak_layers)
    ci_lower = np.percentile(peak_layers, 2.5)
    ci_upper = np.percentile(peak_layers, 97.5)
    
    return median_peak, ci_lower, ci_upper
