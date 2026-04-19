from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from core_utils import StageTimer, make_xgb_classifier, safe_fit_xgb, seed_everything, standardize_fit_transform
from datasets import build_views, ensure_valid_split, load_iscx16, save_topk_features
from layer_decision import CentroidVerifier, apply_layered_decision, calibrate_thresholds, pick_routed_indices, routing_summary
from mumfs import MUMFS1_SparseHashScorer, MUMFS_MultiView_Adaptive, print_view_composition
from tagan_f import FeatureTAGAN_Feedback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def balance_to_target_strict(
    X: np.ndarray,
    y_str: np.ndarray,
    le: LabelEncoder,
    gan: FeatureTAGAN_Feedback | None = None,
    target_count: int = 600,
    gan_epochs: int = 200,
    gan_batch: int = 256,
    min_gen_req: int = 200,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float32)
    y_str = np.asarray(y_str)
    y_enc = le.transform(y_str)
    n_classes = len(le.classes_)
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.int32)
    maxc = int(counts.max()) if counts.size else 0
    eff_target = max(1, min(int(target_count), maxc))

    if gan is not None:
        X_gan_list, y_gan_list = [], []
        for c in range(n_classes):
            idx = np.where(y_enc == c)[0]
            if idx.size == 0:
                continue
            Xc = X[idx]
            if len(Xc) < 80:
                sel = rng.choice(len(Xc), 200, replace=True)
                X_gan_list.append(Xc[sel])
                y_gan_list.append(np.full(200, c, dtype=np.int32))
            else:
                X_gan_list.append(Xc)
                y_gan_list.append(np.full(len(Xc), c, dtype=np.int32))
        X_gan = np.vstack(X_gan_list).astype(np.float32, copy=False)
        y_gan = np.concatenate(y_gan_list).astype(np.int32, copy=False)
        gan.train(X_gan, y_gan, epochs=int(gan_epochs), batch_size=int(gan_batch), lambda_cls=0.5)

    X_out, y_out = [], []
    for c in tqdm(range(n_classes), desc="Balancing", ncols=100):
        idx = np.where(y_enc == c)[0]
        if idx.size == 0:
            continue
        Xc = X[idx]
        if len(Xc) >= eff_target:
            sel = rng.choice(len(Xc), eff_target, replace=False)
            X_out.append(Xc[sel])
            y_out.append(np.full(eff_target, c, dtype=np.int32))
            continue

        X_out.append(Xc)
        y_out.append(np.full(len(Xc), c, dtype=np.int32))
        req = int(eff_target - len(Xc))
        if gan is not None and req >= int(min_gen_req):
            syn = gan.sample(req, c)
        else:
            sel = rng.choice(len(Xc), req, replace=True)
            syn = Xc[sel].astype(np.float32, copy=False)
        X_out.append(syn)
        y_out.append(np.full(req, c, dtype=np.int32))

    Xb = np.vstack(X_out).astype(np.float32, copy=False)
    yb = np.concatenate(y_out).astype(np.int32, copy=False)
    return Xb, yb


def aggressive_balancing(
    X: np.ndarray,
    y_str: np.ndarray,
    gan: FeatureTAGAN_Feedback | None = None,
    target_count: int = 600,
    gan_epochs: int = 300,
    gan_batch: int = 256,
    seed: int = 42,
):
    le = LabelEncoder()
    le.fit(np.asarray(y_str))
    Xb, yb = balance_to_target_strict(
        X=np.asarray(X, dtype=np.float32),
        y_str=np.asarray(y_str),
        le=le,
        gan=gan,
        target_count=int(target_count),
        gan_epochs=int(gan_epochs),
        gan_batch=int(gan_batch),
        seed=int(seed),
    )
    return Xb, le.inverse_transform(yb)


def run_mumfs_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    views_mode: str,
    idx_seq,
    idx_stats,
    idx_fft,
    min_features: int,
    max_features: int,
    fs_limit_tr: int = 20000,
    fs_limit_va: int = 10000,
    scorer_max_samples: int = 8000,
    verbose_scorer: bool = True,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    if len(X_train) > fs_limit_tr:
        sel = rng.choice(len(X_train), fs_limit_tr, replace=False)
        X_fs_tr, y_fs_tr = X_train[sel], y_train[sel]
    else:
        X_fs_tr, y_fs_tr = X_train, y_train
    if len(X_valid) > fs_limit_va:
        sel = rng.choice(len(X_valid), fs_limit_va, replace=False)
        X_fs_va, y_fs_va = X_valid[sel], y_valid[sel]
    else:
        X_fs_va, y_fs_va = X_valid, y_valid

    views_tr, view_names, view_global = build_views(X_fs_tr, idx_seq, idx_stats, idx_fft, views_mode)
    views_va, _, _ = build_views(X_fs_va, idx_seq, idx_stats, idx_fft, views_mode)
    scorer = MUMFS1_SparseHashScorer(
        bits=12,
        knn_k=10,
        num_iter=12,
        alpha=1.0,
        beta=0.1,
        mu=0.05,
        eta=2.0,
        lambda0=1.0,
        rho=1.05,
        max_samples=scorer_max_samples,
        ridge=1e-4,
        rp_dim=64,
        seed=seed,
        verbose=verbose_scorer,
    )
    selector = MUMFS_MultiView_Adaptive(
        scorer=scorer,
        min_features=int(min_features),
        max_features_limit=int(max_features),
        k_mode="valid",
        quick_metric="macro_f1",
        within_delta=0.01,
        k_candidates=tuple(sorted(set([int(min_features), 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, int(max_features)]))),
    )
    selector.fit(views_tr, y_fs_tr, view_global, views_va, y_fs_va, view_names=view_names)
    print_view_composition("Feature Selection", selector.view_composition_, len(selector.selected_indices_))
    return selector


def train_l1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    le: LabelEncoder,
    use_gan: bool,
    gan_target_count: int = 900,
    gan_epochs: int = 300,
    gan_batch: int = 256,
    recall_target: float = 0.98,
    min_thr: float = 0.03,
    calib_min_support: int = 10,
    calib_fallback_thr: float = 0.15,
):
    n_classes = len(le.classes_)
    y_train_str = le.inverse_transform(y_train)

    if use_gan:
        tagan = FeatureTAGAN_Feedback(n_classes, X_train.shape[1], device=DEVICE)
        X_bal, y_bal_str = aggressive_balancing(
            X_train,
            y_train_str,
            tagan,
            target_count=int(gan_target_count),
            gan_epochs=int(gan_epochs),
            gan_batch=int(gan_batch),
        )
        y_bal = le.transform(y_bal_str)
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        maxc = float(np.max(counts) + 1e-9)
        w_map = (maxc / (counts + 1e-9)) ** 0.9
        sample_w = w_map[y_bal].astype(np.float32)
    else:
        X_bal, y_bal = X_train, y_train
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        maxc = float(np.max(counts) + 1e-9)
        w_map = (maxc / (counts + 1e-9)) ** 0.8
        sample_w = w_map[y_bal].astype(np.float32)

    clf = make_xgb_classifier(
        device=DEVICE,
        n_classes=n_classes,
        n_estimators=6000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1e-3,
        gamma=0.1,
    )
    safe_fit_xgb(
        clf,
        X_bal,
        y_bal,
        sample_weight=sample_w,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=120,
        verbose=False,
    )
    thr_map = calibrate_thresholds(
        clf,
        X_valid,
        y_valid,
        le,
        recall_target=float(recall_target),
        min_thr=float(min_thr),
        max_thr=0.8,
        min_support=int(calib_min_support),
        fallback_thr=float(calib_fallback_thr),
    )
    return clf, thr_map


def train_l2(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, le: LabelEncoder):
    n_classes = len(le.classes_)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    maxc = float(np.max(counts) + 1e-9)
    sample_w = ((maxc / (counts + 1e-9)) ** 1.2)[y_train].astype(np.float32)

    clf = make_xgb_classifier(
        device=DEVICE,
        n_classes=n_classes,
        n_estimators=8000,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=1e-3,
        gamma=0.05,
    )
    safe_fit_xgb(
        clf,
        X_train,
        y_train,
        sample_weight=sample_w,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=120,
        verbose=False,
    )
    X_tr_s, stats = standardize_fit_transform(X_train)
    centroid = CentroidVerifier().fit(X_tr_s, y_train)
    return clf, centroid, stats


@dataclass
class ISCX16Config:
    csv_file: str
    views_mode: str = "all"
    use_mumfs: bool = True
    use_gan: bool = True
    use_l2: bool = True
    use_centroid_gate: bool = True
    l1_min_k: int = 20
    l1_max_k: int = 128
    l2_min_k: int = 20
    l2_max_k: int = 128
    gan_target_count: int = 900
    gan_epochs: int = 300
    gan_batch: int = 128
    recall_target: float = 0.92
    min_thr: float = 0.03
    calib_min_support: int = 10
    calib_fallback_thr: float = 0.15
    min_route_ratio: float = 0.20
    max_route_ratio: Optional[float] = None
    route_rank_by: str = "margin"
    centroid_accept_thr: float = 0.90
    topk_csv_prefix: str = "iscx16"


def run_iscx16_pipeline(cfg: ISCX16Config):
    print("=" * 88)
    print(
        "ISCX-VPN 16 | "
        f"views={cfg.views_mode} | mumfs={cfg.use_mumfs} | gan={cfg.use_gan} | "
        f"l2={cfg.use_l2} | centroid_gate={cfg.use_centroid_gate}"
    )
    print("=" * 88)

    with StageTimer("Load data"):
        X, y_str, splits, (idx_seq, idx_stats, idx_fft), feat_names = load_iscx16(cfg.csv_file)
        le = LabelEncoder()
        y = le.fit_transform(y_str)
        m_tr, m_va, m_te = ensure_valid_split(splits, seed=42, fallback_ratio=0.1)
        X_tr, X_va, X_te = X[m_tr], X[m_va], X[m_te]
        y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
        print(f"Samples={len(X)} | Classes={len(le.classes_)}")
        print(f"Split: train={len(X_tr)} valid={len(X_va)} test={len(X_te)}")
        print(f"Stats={len(idx_stats)} Seq={len(idx_seq)} FFT={len(idx_fft)}")

    with StageTimer("Phase1: L1 Feature Selection"):
        if cfg.use_mumfs:
            sel1 = run_mumfs_feature_selection(
                X_tr,
                y_tr,
                X_va,
                y_va,
                views_mode=cfg.views_mode,
                idx_seq=idx_seq,
                idx_stats=idx_stats,
                idx_fft=idx_fft,
                min_features=cfg.l1_min_k,
                max_features=min(cfg.l1_max_k, X_tr.shape[1]),
                verbose_scorer=True,
            )
            save_topk_features(sel1, feat_names, f"{cfg.topk_csv_prefix}_l1_top_features.csv", topk=30)
            X_tr_l1 = sel1.transform(X_tr).astype(np.float32, copy=False)
            X_va_l1 = sel1.transform(X_va).astype(np.float32, copy=False)
            X_te_l1 = sel1.transform(X_te).astype(np.float32, copy=False)
        else:
            X_tr_l1, X_va_l1, X_te_l1 = X_tr, X_va, X_te

    with StageTimer("Phase2: Train L1"):
        clf_l1, thr_map = train_l1(
            X_tr_l1,
            y_tr,
            X_va_l1,
            y_va,
            le,
            use_gan=cfg.use_gan,
            gan_target_count=cfg.gan_target_count,
            gan_epochs=cfg.gan_epochs,
            gan_batch=cfg.gan_batch,
            recall_target=cfg.recall_target,
            min_thr=cfg.min_thr,
            calib_min_support=cfg.calib_min_support,
            calib_fallback_thr=cfg.calib_fallback_thr,
        )

    clf_l2 = None
    centroid = None
    stats_l2 = None
    X_te_l2 = None
    if cfg.use_l2:
        with StageTimer("Phase3: Train L2 Expert"):
            if cfg.use_mumfs:
                sel2 = run_mumfs_feature_selection(
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    views_mode=cfg.views_mode,
                    idx_seq=idx_seq,
                    idx_stats=idx_stats,
                    idx_fft=idx_fft,
                    min_features=cfg.l2_min_k,
                    max_features=min(cfg.l2_max_k, X_tr.shape[1]),
                    verbose_scorer=False,
                )
                save_topk_features(sel2, feat_names, f"{cfg.topk_csv_prefix}_l2_top_features.csv", topk=30)
                X_tr_l2 = sel2.transform(X_tr).astype(np.float32, copy=False)
                X_va_l2 = sel2.transform(X_va).astype(np.float32, copy=False)
                X_te_l2 = sel2.transform(X_te).astype(np.float32, copy=False)
            else:
                X_tr_l2, X_va_l2, X_te_l2 = X_tr, X_va, X_te
            clf_l2, centroid, stats_l2 = train_l2(X_tr_l2, y_tr, X_va_l2, y_va, le)

    with StageTimer("Phase4: Inference & Routing"):
        probs1 = clf_l1.predict_proba(X_te_l1.astype(np.float32, copy=False))
        pred1 = np.argmax(probs1, axis=1)
        final = pred1.copy()
        info = {"total": len(final), "routed": 0, "base_routed": 0, "routed_ratio": 0.0}
        if cfg.use_l2 and clf_l2 is not None and X_te_l2 is not None:
            idx_route, base_cnt = pick_routed_indices(
                probs=probs1,
                pred=pred1,
                thr_map=thr_map,
                min_route_ratio=cfg.min_route_ratio,
                max_route_ratio=cfg.max_route_ratio,
                rank_by=cfg.route_rank_by,
            )
            info = routing_summary(len(final), idx_route, base_cnt)
            final = apply_layered_decision(
                probs_l1=probs1,
                clf_l2=clf_l2,
                X_l2=X_te_l2,
                routed_idx=idx_route,
                final_pred=final,
                centroid=(centroid if cfg.use_centroid_gate else None),
                centroid_stats=(stats_l2 if cfg.use_centroid_gate else None),
                centroid_accept_thr=cfg.centroid_accept_thr,
            )

        acc = accuracy_score(y_te, final)
        macro_f1 = f1_score(y_te, final, average="macro")
        print(f"ACC={acc:.4f} | MacroF1={macro_f1:.4f}")
        print(f"Routing: {info}")
        print("\n" + classification_report(le.inverse_transform(y_te), le.inverse_transform(final), digits=4))
        print("[Confusion Matrix]")
        print(confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(final)))

    del clf_l1
    if clf_l2 is not None:
        del clf_l2
    torch.cuda.empty_cache()
    gc.collect()
    return {"acc": float(acc), "macro_f1": float(macro_f1), "routing": info}


def main():
    parser = argparse.ArgumentParser(description="Run the public ISCX shortcut-resistant ETC pipeline.")
    parser.add_argument("--csv-file", required=True, help="Path to the extracted ISCX CSV.")
    parser.add_argument("--views-mode", default="all", choices=["all", "seq", "stats", "fft", "seq_stats", "seq_fft", "stats_fft"])
    parser.add_argument("--no-mumfs", action="store_true")
    parser.add_argument("--no-gan", action="store_true")
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument("--no-centroid-gate", action="store_true")
    parser.add_argument("--gan-target-count", type=int, default=900)
    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--gan-batch", type=int, default=128)
    parser.add_argument("--min-route-ratio", type=float, default=0.20)
    parser.add_argument("--route-rank-by", default="margin", choices=["margin", "confidence"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    cfg = ISCX16Config(
        csv_file=args.csv_file,
        views_mode=args.views_mode,
        use_mumfs=not args.no_mumfs,
        use_gan=not args.no_gan,
        use_l2=not args.no_l2,
        use_centroid_gate=not args.no_centroid_gate,
        gan_target_count=int(args.gan_target_count),
        gan_epochs=int(args.gan_epochs),
        gan_batch=int(args.gan_batch),
        min_route_ratio=float(args.min_route_ratio),
        route_rank_by=str(args.route_rank_by),
    )
    run_iscx16_pipeline(cfg)


if __name__ == "__main__":
    main()
