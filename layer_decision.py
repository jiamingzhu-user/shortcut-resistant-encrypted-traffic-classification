from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from core_utils import standardize_transform


class CentroidVerifier:
    def __init__(self):
        self.centroids_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        centroids = [X[y == c].mean(axis=0) for c in classes]
        self.classes_ = classes
        self.centroids_ = np.vstack(centroids).astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        C = np.asarray(self.centroids_, dtype=np.float32)
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(C * C, axis=1, keepdims=True).T
        dist = x2 + c2 - 2.0 * (X @ C.T)
        idx = np.argmin(dist, axis=1)
        return self.classes_[idx]


def calibrate_thresholds(
    clf,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    le: LabelEncoder,
    recall_target: float = 0.98,
    min_thr: float = 0.03,
    max_thr: float = 0.8,
    thr_scale: float = 1.0,
    min_support: int = 10,
    fallback_thr: float = 0.15,
) -> Dict[int, float]:
    probs = clf.predict_proba(np.asarray(X_calib, dtype=np.float32))
    thresholds: Dict[int, float] = {}
    print(
        f"[Calibration] recall_target={recall_target:.2f} "
        f"min_thr={min_thr:.2f} min_support={min_support} fallback_thr={fallback_thr:.2f}"
    )
    for cls_idx, cls_name in enumerate(le.classes_):
        true_mask = y_calib == cls_idx
        n_true = int(np.sum(true_mask))
        if n_true == 0:
            thresholds[cls_idx] = float(min_thr)
            print(f"  {cls_name}: no valid support, use {thresholds[cls_idx]:.4f}")
            continue
        if n_true < int(min_support):
            cutoff = max(min_thr, min(float(fallback_thr) * thr_scale, max_thr))
            thresholds[cls_idx] = cutoff
            print(f"  {cls_name}: support={n_true}, fallback={cutoff:.4f}")
            continue
        target_probs = probs[true_mask, cls_idx]
        cutoff = float(np.quantile(target_probs, 1.0 - recall_target))
        cutoff = max(min_thr, min(cutoff * thr_scale, max_thr))
        thresholds[cls_idx] = cutoff
    return thresholds


def top2_margin(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    part = np.partition(probs, kth=-2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return (top1 - top2).astype(np.float32)


def pick_routed_indices(
    probs: np.ndarray,
    pred: np.ndarray,
    thr_map: Dict[int, float],
    min_route_ratio: float = 0.10,
    max_route_ratio: Optional[float] = None,
    rank_by: str = "margin",
) -> Tuple[np.ndarray, int]:
    probs = np.asarray(probs, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.int32)
    n = int(len(pred))
    conf = probs[np.arange(n), pred]
    thr_vec = np.array([float(thr_map.get(int(c), 0.3)) for c in pred], dtype=np.float32)
    base = conf < thr_vec
    score = top2_margin(probs) if rank_by == "margin" else conf

    routed = base.copy()
    n_min = int(np.ceil(max(0.0, float(min_route_ratio)) * n))
    if routed.sum() < n_min:
        need = n_min - int(routed.sum())
        cand = np.where(~routed)[0]
        pick = cand[np.argsort(score[cand])[:need]]
        routed[pick] = True

    if max_route_ratio is not None:
        n_max = int(np.floor(max(0.0, float(max_route_ratio)) * n))
        if routed.sum() > n_max and n_max > 0:
            ridx = np.where(routed)[0]
            keep = ridx[np.argsort(score[ridx])[:n_max]]
            routed[:] = False
            routed[keep] = True
    return np.where(routed)[0], int(base.sum())


def apply_layered_decision(
    probs_l1: np.ndarray,
    clf_l2,
    X_l2: np.ndarray,
    routed_idx: np.ndarray,
    *,
    final_pred: np.ndarray,
    centroid: Optional[CentroidVerifier] = None,
    centroid_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    centroid_accept_thr: float = 0.90,
) -> np.ndarray:
    if len(routed_idx) == 0:
        return final_pred

    sub = np.asarray(X_l2[routed_idx], dtype=np.float32)
    probs2 = clf_l2.predict_proba(sub)
    pred2 = np.argmax(probs2, axis=1)
    if centroid is None or centroid_stats is None:
        final_pred[routed_idx] = pred2
        return final_pred

    sub_s = standardize_transform(sub, centroid_stats)
    predm = centroid.predict(sub_s)
    for k, ii in enumerate(routed_idx):
        px = int(pred2[k])
        pm = int(predm[k])
        if px == pm or float(np.max(probs2[k])) > float(centroid_accept_thr):
            final_pred[ii] = px
    return final_pred


def routing_summary(total: int, routed_idx: np.ndarray, base_count: int) -> Dict[str, Any]:
    return {
        "total": int(total),
        "routed": int(len(routed_idx)),
        "routed_ratio": float(len(routed_idx) / max(1, total)),
        "base_routed": int(base_count),
    }
