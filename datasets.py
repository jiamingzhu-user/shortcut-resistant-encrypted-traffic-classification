from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _make_stratified_splits(
    y_raw: np.ndarray,
    seed: int = 42,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> np.ndarray:
    y_raw = np.asarray(y_raw)
    rng = np.random.RandomState(seed)
    splits = np.array(["train"] * len(y_raw), dtype=object)
    for cls in np.unique(y_raw):
        idx = np.where(y_raw == cls)[0]
        rng.shuffle(idx)
        m = len(idx)
        if m < 5:
            continue
        n_te = min(max(1, int(round(m * test_ratio))), m - 2)
        n_va = min(max(1, int(round(m * valid_ratio))), m - n_te - 1)
        splits[idx[:n_te]] = "test"
        splits[idx[n_te:n_te + n_va]] = "valid"
    return splits.astype(str)


def load_iscx16(csv_file: str):
    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        raise ValueError("ISCX CSV must contain a 'label' column.")

    y = df["label"].values
    if "split" in df.columns:
        splits = df["split"].astype(str).str.lower().values
    else:
        splits = np.char.lower(_make_stratified_splits(y, seed=42, valid_ratio=0.15, test_ratio=0.15))

    feat_df = df.drop(columns=["label", "split"], errors="ignore").copy()
    feat_names = list(feat_df.columns)

    for col in feat_names:
        series = feat_df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) >= max(1, int(0.95 * len(numeric))):
            feat_df[col] = numeric.fillna(0.0)
        else:
            encoder = LabelEncoder()
            feat_df[col] = encoder.fit_transform(series.astype(str).fillna("UNK"))

    X_raw = feat_df.values.astype(np.float32, copy=False)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    log_mask = np.ones(len(feat_names), dtype=bool)
    for i, name in enumerate(feat_names):
        lower = str(name).lower()
        if lower.startswith("seq_dir_"):
            log_mask[i] = False
    X = X_raw.copy()
    X[:, log_mask] = np.sign(X_raw[:, log_mask]) * np.log1p(np.abs(X_raw[:, log_mask]))

    idx_seq, idx_fft, idx_stats = [], [], []
    for i, name in enumerate(feat_names):
        lower = str(name).lower()
        if lower.startswith("seq_"):
            idx_seq.append(i)
        elif lower.startswith("fft_"):
            idx_fft.append(i)
        else:
            idx_stats.append(i)

    return X.astype(np.float32), y, splits, (idx_seq, idx_stats, idx_fft), feat_names


def ensure_valid_split(splits: np.ndarray, seed: int = 42, fallback_ratio: float = 0.1):
    splits = np.asarray(splits).astype(str)
    mask_train = splits == "train"
    mask_valid = np.isin(splits, ["valid", "val", "validation"])
    mask_test = splits == "test"
    if mask_valid.sum() > 0:
        return mask_train, mask_valid, mask_test

    rng = np.random.RandomState(seed)
    tr_idx = np.where(mask_train)[0]
    rng.shuffle(tr_idx)
    cut = int(len(tr_idx) * (1.0 - fallback_ratio))
    tr2, va2 = tr_idx[:cut], tr_idx[cut:]
    mask_train2 = np.zeros_like(mask_train, dtype=bool)
    mask_valid2 = np.zeros_like(mask_train, dtype=bool)
    mask_train2[tr2] = True
    mask_valid2[va2] = True
    return mask_train2, mask_valid2, mask_test


def build_views(
    X: np.ndarray,
    idx_seq: List[int],
    idx_stats: List[int],
    idx_fft: List[int],
    mode: str = "all",
) -> Tuple[List[np.ndarray], List[str], List[List[int]]]:
    mode = str(mode).lower()
    views: List[np.ndarray] = []
    names: List[str] = []
    globals_: List[List[int]] = []

    use_seq = mode in ("all", "seq", "seq_stats", "seq_fft")
    use_stats = mode in ("all", "stats", "seq_stats", "stats_fft")
    use_fft = mode in ("all", "fft", "seq_fft", "stats_fft")

    if use_stats and idx_stats:
        views.append(X[:, idx_stats])
        names.append("Stats")
        globals_.append(idx_stats)
    if use_seq and idx_seq:
        views.append(X[:, idx_seq])
        names.append("Sequence")
        globals_.append(idx_seq)
    if use_fft and idx_fft:
        views.append(X[:, idx_fft])
        names.append("FFT")
        globals_.append(idx_fft)

    if not views:
        all_idx = list(range(X.shape[1]))
        views = [X[:, all_idx]]
        names = ["All"]
        globals_ = [all_idx]
    return views, names, globals_


def save_topk_features(selector, feat_names: List[str], save_path: str, topk: int = 30) -> None:
    if selector is None or getattr(selector, "selected_ranked_", None) is None:
        return
    rows = []
    for rank, (gidx, score, view_name) in enumerate(selector.selected_ranked_[:topk], start=1):
        fname = feat_names[gidx] if 0 <= gidx < len(feat_names) else f"f{gidx}"
        rows.append(
            {
                "rank": rank,
                "global_idx": int(gidx),
                "feature_name": fname,
                "score": float(score),
                "view": view_name,
            }
        )
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
