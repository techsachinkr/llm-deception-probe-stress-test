"""
Probe training: linear, multi-dimensional, and MLP probes (§4.4).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from src.metrics import auroc, compute_all_metrics

logger = logging.getLogger(__name__)


# ── Linear Probe (§4.5.1) ────────────────────────────────────────────

class LinearProbe:
    """
    Logistic regression probe with L2 regularization.
    C selected from {0.01, 0.1, 1.0, 10.0} via inner CV (§4.5.1).
    """

    def __init__(self, C_values: List[float] = None, inner_cv_folds: int = 3, seed: int = 42):
        self.C_values = C_values or [0.01, 0.1, 1.0, 10.0]
        self.inner_cv_folds = inner_cv_folds
        self.seed = seed
        self.scaler = StandardScaler()
        self.clf = None
        self.best_C = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit with inner CV for C selection."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf = LogisticRegressionCV(
            Cs=self.C_values,
            cv=self.inner_cv_folds,
            scoring="roc_auc",
            solver="lbfgs",
            max_iter=5000,
            random_state=self.seed,
            penalty="l2",
        )
        self.clf.fit(X_scaled, y)
        self.best_C = self.clf.C_[0]
        logger.debug(f"LinearProbe: best C={self.best_C}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return auroc(y, self.predict_proba(X))

    @property
    def coef_(self) -> np.ndarray:
        """Probe weight vector (in original space)."""
        # Un-scale the coefficients: w_original = w_scaled / scale
        return self.clf.coef_[0] / self.scaler.scale_

    @property
    def direction(self) -> np.ndarray:
        """Unit probe direction."""
        w = self.coef_
        return w / np.linalg.norm(w)


# ── Multi-Dimensional Probe (§4.5.2) ─────────────────────────────────

class MultiDimProbe:
    """
    Logistic regression on top-k PCA components (§4.5.2).
    Tests whether deception signal is multi-dimensional.
    """

    def __init__(self, k: int, C_values: List[float] = None, seed: int = 42):
        self.k = k
        self.C_values = C_values or [0.01, 0.1, 1.0, 10.0]
        self.seed = seed
        self.pca = PCA(n_components=k, random_state=seed)
        self.scaler = StandardScaler()
        self.clf = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_pca = self.pca.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_pca)
        self.clf = LogisticRegressionCV(
            Cs=self.C_values,
            cv=3,
            scoring="roc_auc",
            solver="lbfgs",
            max_iter=5000,
            random_state=self.seed,
        )
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return auroc(y, self.predict_proba(X))

    @property
    def variance_explained(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


# ── MLP Probe (§4.5.3) ───────────────────────────────────────────────

class MLPProbeNet(nn.Module):
    """2-layer MLP probe network."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPProbe:
    """
    2-layer MLP probe for nonlinear upper bound (§4.5.3).
    Hidden dim 256, ReLU, dropout 0.3, early stopping on val AUROC.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        torch.manual_seed(seed)
        self.model = MLPProbeNet(input_dim, hidden_dim, dropout).to(self.device)
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train with early stopping on validation AUROC."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_v = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auroc = 0.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Mini-batch training
            batch_size = 256
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0
            
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X_t[idx])
                loss = criterion(logits, y_t[idx])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_auroc = auroc(y_val, val_probs)
                
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.debug(f"Early stopping at epoch {epoch}, best val AUROC={best_val_auroc:.4f}")
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return auroc(y, self.predict_proba(X))


# ── Text Baseline (for Black-to-White Boost) ─────────────────────────

class TextBaseline:
    """TF-IDF + Logistic Regression text-only classifier (§4.5.4)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clf = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(self, texts: List[str], y: np.ndarray):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

    def score(self, texts: List[str], y: np.ndarray) -> float:
        return auroc(y, self.predict_proba(texts))


# ── Probe Training Orchestration ──────────────────────────────────────

def train_probes_all_layers(
    activations: np.ndarray,    # (n, n_layers, d_model)
    labels: np.ndarray,         # (n,)
    config,
    val_activations: np.ndarray = None,
    val_labels: np.ndarray = None,
    min_layer_frac: float = 0.3,
) -> Dict:
    """
    Train linear probes at every layer and find the best layer.
    Also trains multi-dim and MLP probes at the best layer.
    
    min_layer_frac: Skip early layers (default 30% of depth) to avoid
    the system prompt confound where early layers trivially separate
    honest/deceptive based on prompt encoding, not deception content.
    """
    n_samples, n_layers, d_model = activations.shape
    min_layer = int(n_layers * min_layer_frac)
    
    results = {
        "layer_aurocs": np.zeros(n_layers),
        "layer_probes": {},
        "best_layer": min_layer,
        "best_auroc": 0.0,
        "min_layer": min_layer,
    }
    
    # 1. Train linear probes at layers >= min_layer
    logger.info(f"Training linear probes across layers {min_layer}-{n_layers-1} "
                f"(skipping first {min_layer} layers to avoid prompt confound)...")
    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = LinearProbe(
            C_values=config.probe.C_values,
            inner_cv_folds=config.probe.inner_cv_folds,
            seed=config.probe.seeds[0],
        )
        probe.fit(X, labels)
        
        if val_activations is not None:
            X_val = val_activations[:, layer, :]
            layer_auroc = probe.score(X_val, val_labels)
        else:
            layer_auroc = probe.score(X, labels)
        
        results["layer_aurocs"][layer] = layer_auroc
        results["layer_probes"][layer] = probe
        
        # Only consider layers above min_layer for best layer selection
        if layer >= min_layer and layer_auroc > results["best_auroc"]:
            results["best_auroc"] = layer_auroc
            results["best_layer"] = layer
    
    best_layer = results["best_layer"]
    logger.info(f"Best layer: {best_layer} (AUROC={results['best_auroc']:.4f})")
    
    # 2. Train multi-dimensional probes at best layer (§4.5.2)
    X_best = activations[:, best_layer, :]
    X_val_best = val_activations[:, best_layer, :] if val_activations is not None else None
    
    results["multi_dim"] = {}
    for k in config.probe.k_dims:
        if k > min(n_samples, d_model):
            continue
        mdp = MultiDimProbe(k=k, seed=config.probe.seeds[0])
        mdp.fit(X_best, labels)
        if X_val_best is not None:
            md_auroc = mdp.score(X_val_best, val_labels)
        else:
            md_auroc = mdp.score(X_best, labels)
        results["multi_dim"][k] = {"probe": mdp, "auroc": md_auroc}
        logger.info(f"  Multi-dim k={k}: AUROC={md_auroc:.4f}")
    
    # 3. Train MLP probe at best layer (§4.5.3)
    mlp = MLPProbe(
        input_dim=d_model,
        hidden_dim=config.probe.mlp_hidden_dim,
        dropout=config.probe.mlp_dropout,
        epochs=config.probe.mlp_epochs,
        lr=config.probe.mlp_lr,
        patience=config.probe.mlp_patience,
        seed=config.probe.seeds[0],
    )
    mlp.fit(X_best, labels, X_val_best, val_labels)
    if X_val_best is not None:
        mlp_auroc = mlp.score(X_val_best, val_labels)
    else:
        mlp_auroc = mlp.score(X_best, labels)
    results["mlp"] = {"probe": mlp, "auroc": mlp_auroc}
    logger.info(f"  MLP probe: AUROC={mlp_auroc:.4f}")
    
    return results
