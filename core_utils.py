from __future__ import annotations

import inspect
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import xgboost as xgb


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class StageTimer:
    name: str
    t0: float = 0.0

    def __enter__(self):
        _sync_cuda()
        self.t0 = time.perf_counter()
        print(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync_cuda()
        elapsed = time.perf_counter() - self.t0
        print(f"[DONE ] {self.name} | elapsed={elapsed:.2f}s")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_fit_transform(X_train: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    X_train = np.asarray(X_train, dtype=np.float32)
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma_floor = max(1e-6, float(np.mean(sigma)) * 1e-6)
    sigma = np.maximum(sigma, sigma_floor)
    return (X_train - mu) / sigma, (mu, sigma)


def standardize_transform(X: np.ndarray, stats: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu, sigma = stats
    return (X - mu) / sigma


def make_xgb_classifier(device: str, n_classes: Optional[int] = None, **kwargs):
    params = dict(kwargs)
    params.setdefault("verbosity", 0)
    params.setdefault("n_jobs", max(1, (os.cpu_count() or 4) // 2))

    if "objective" not in params:
        if n_classes is not None and int(n_classes) <= 2:
            params["objective"] = "binary:logistic"
            params.setdefault("eval_metric", "logloss")
            params.pop("num_class", None)
        else:
            params["objective"] = "multi:softprob"
            params.setdefault("eval_metric", "mlogloss")
            if n_classes is not None:
                params["num_class"] = int(n_classes)

    if device == "cuda":
        try:
            return xgb.XGBClassifier(device="cuda", tree_method="hist", **params)
        except TypeError:
            try:
                return xgb.XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor", **params)
            except TypeError:
                return xgb.XGBClassifier(tree_method="gpu_hist", **params)

    try:
        return xgb.XGBClassifier(tree_method="hist", predictor="cpu_predictor", **params)
    except TypeError:
        return xgb.XGBClassifier(tree_method="hist", **params)


def safe_fit_xgb(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight=None,
    eval_set=None,
    early_stopping_rounds: int = 80,
    verbose: bool = False,
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    fit_sig = inspect.signature(clf.fit)
    kwargs = {
        "X": X,
        "y": y,
        "sample_weight": sample_weight,
        "eval_set": eval_set,
        "verbose": verbose,
    }
    if "early_stopping_rounds" in fit_sig.parameters and eval_set is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    clf.fit(**kwargs)
    return clf
