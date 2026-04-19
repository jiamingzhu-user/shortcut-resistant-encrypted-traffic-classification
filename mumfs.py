from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score

from core_utils import standardize_fit_transform, standardize_transform
from layer_decision import CentroidVerifier


def proj_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    v = v.astype(np.float64, copy=False)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, z / v.size, dtype=np.float64)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norm


def constructW_knn_cosine(fea: np.ndarray, k: int = 10, rp_dim: int = 64, seed: int = 42) -> sp.csr_matrix:
    rng = np.random.RandomState(seed)
    X = fea.astype(np.float32, copy=False)
    n, d = X.shape
    k = int(min(k, max(1, n - 1)))

    if d > rp_dim:
        proj = rng.randn(d, rp_dim).astype(np.float32) / np.sqrt(rp_dim)
        Xp = X @ proj
    else:
        Xp = X

    tree = cKDTree(Xp)
    _, idx = tree.query(Xp, k=k + 1)
    idx = idx[:, 1:]

    Xn = _normalize_rows(X)
    neigh = Xn[idx]
    sim = np.sum(neigh * Xn[:, None, :], axis=2)
    sim = np.clip(sim, 0.0, None).astype(np.float32)

    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = idx.reshape(-1).astype(np.int32)
    data = sim.reshape(-1).astype(np.float32)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W.maximum(W.T).tocsr()


def _binarize_ge0(x: np.ndarray) -> np.ndarray:
    return (x >= 0).astype(np.float32)


class MUMFS1_SparseHashScorer:
    def __init__(
        self,
        bits: int = 12,
        knn_k: int = 10,
        num_iter: int = 12,
        alpha: float = 1.0,
        beta: float = 0.1,
        mu: float = 0.05,
        eta: float = 2.0,
        lambda0: float = 1.0,
        rho: float = 1.05,
        max_samples: int = 8000,
        ridge: float = 1e-4,
        rp_dim: int = 64,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.bits = int(bits)
        self.knn_k = int(knn_k)
        self.num_iter = int(num_iter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu = float(mu)
        self.eta = float(eta)
        if self.eta <= 1.0:
            raise ValueError("MUMFS eta must be > 1 for the inverse-discrepancy view-weight update.")
        self.lambda0 = float(lambda0)
        self.rho = float(rho)
        self.max_samples = int(max_samples)
        self.ridge = float(ridge)
        self.rp_dim = int(rp_dim)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def fit_score_multi(self, X_views: List[np.ndarray]) -> List[np.ndarray]:
        rng = np.random.RandomState(self.seed)
        n = X_views[0].shape[0]
        for xv in X_views:
            if xv.shape[0] != n:
                raise ValueError("All views must share the same sample count.")

        if n > self.max_samples:
            sel = rng.choice(n, self.max_samples, replace=False)
            Xv = [x[sel].astype(np.float32, copy=False) for x in X_views]
        else:
            Xv = [x.astype(np.float32, copy=False) for x in X_views]

        n = Xv[0].shape[0]
        dims = [x.shape[1] for x in Xv]
        total_dim = int(sum(dims))
        graphs = [constructW_knn_cosine(x, k=self.knn_k, rp_dim=self.rp_dim, seed=self.seed + 17 * i) for i, x in enumerate(Xv)]
        G = sum(graphs) * (1.0 / len(graphs))
        G = ((G + G.T) * 0.5).tocsr()

        X = np.concatenate(Xv, axis=1).astype(np.float32, copy=False)
        B = _binarize_ge0(rng.randn(X.shape[0], self.bits).astype(np.float32))
        Z = B.copy()
        M = np.zeros((X.shape[0], self.bits), dtype=np.float32)
        lam = float(self.lambda0)
        P = rng.randn(total_dim, self.bits).astype(np.float32) * 0.01
        H_diag = np.ones(total_dim, dtype=np.float32)
        XtX = (X.T @ X).astype(np.float32)

        for _ in range(self.num_iter):
            temp = []
            for Av in graphs:
                diff = G - Av
                temp.append(float(diff.multiply(diff).sum()) + 1e-12)
            temp = np.asarray(temp, dtype=np.float64)
            alpha_v = temp ** (-1.0 / (self.eta - 1.0))
            alpha_v = alpha_v / (np.sum(alpha_v) + 1e-12)
            fusion_weights = alpha_v ** self.eta
            fusion_weights = fusion_weights / (np.sum(fusion_weights) + 1e-12)

            rows_acc, cols_acc, data_acc = [], [], []
            B_u = B >= 0.5
            for i in range(X.shape[0]):
                acc: Dict[int, float] = {}
                for v_idx, Av in enumerate(graphs):
                    start, end = Av.indptr[i], Av.indptr[i + 1]
                    js = Av.indices[start:end]
                    ws = Av.data[start:end]
                    for j, w in zip(js, ws):
                        acc[int(j)] = acc.get(int(j), 0.0) + float(fusion_weights[v_idx]) * float(w)
                if not acc:
                    continue
                cand = np.array(list(acc.keys()), dtype=np.int32)
                ai = np.array([acc[int(j)] for j in cand], dtype=np.float32)
                di = np.sum(B_u[cand] != B_u[i], axis=1).astype(np.float32)
                xi = proj_simplex((ai - 0.5 * self.mu * di).astype(np.float64), z=1.0).astype(np.float32)
                keep = xi > 1e-12
                if np.any(keep):
                    rows_acc.append(np.full(np.sum(keep), i, dtype=np.int32))
                    cols_acc.append(cand[keep])
                    data_acc.append(xi[keep])

            if rows_acc:
                rows = np.concatenate(rows_acc)
                cols = np.concatenate(cols_acc)
                data = np.concatenate(data_acc)
                G = sp.csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]), dtype=np.float32)
            else:
                G = sp.csr_matrix((X.shape[0], X.shape[0]), dtype=np.float32)
            G = ((G + G.T) * 0.5).tocsr()
            G.setdiag(0.0)
            G.eliminate_zeros()

            D = np.array(G.sum(axis=1)).reshape(-1).astype(np.float32)
            rhs = (self.alpha * (X.T @ B)).astype(np.float32)
            A_mat = (self.alpha * XtX).astype(np.float32, copy=True)
            idx = np.arange(total_dim)
            A_mat[idx, idx] += (self.beta * H_diag + self.ridge).astype(np.float32)
            try:
                P = np.linalg.solve(A_mat, rhs).astype(np.float32)
            except np.linalg.LinAlgError:
                P = (np.linalg.pinv(A_mat) @ rhs).astype(np.float32)

            row_norm = np.sqrt(np.sum(P * P, axis=1) + 1e-9).astype(np.float32)
            H_diag = 0.5 / row_norm

            L = sp.diags(D, format="csr") - G
            H_G = ((2.0 * self.alpha + lam) * sp.eye(n, format="csr") + 2.0 * self.mu * L).tocsr()
            rhs_B = (2.0 * self.alpha * (X @ P) + lam * (Z - M)).astype(np.float32)
            B = np.column_stack([spsolve(H_G, rhs_B[:, b]) for b in range(self.bits)]).astype(np.float32)
            Z = (B + M >= 0.5).astype(np.float32)
            M = (M + B - Z).astype(np.float32)
            lam = float(lam * self.rho)

        scores_all = np.sqrt(np.sum(P * P, axis=1) + 1e-12).astype(np.float32)
        out = []
        offset = 0
        for dv in dims:
            out.append(scores_all[offset:offset + dv])
            offset += dv
        return out


class MUMFS_MultiView_Adaptive:
    def __init__(
        self,
        scorer,
        min_features: int = 10,
        max_features_limit: int = 600,
        k_mode: str = "valid",
        k_candidates: Sequence[int] = (20, 40, 60, 80, 120, 160, 200),
        quick_metric: str = "macro_f1",
        within_delta: float = 0.01,
    ):
        self.scorer = scorer
        self.min_k = int(min_features)
        self.max_k = int(max_features_limit)
        self.k_mode = str(k_mode)
        self.k_candidates = tuple(int(x) for x in k_candidates)
        self.quick_metric = str(quick_metric)
        self.within_delta = float(within_delta)
        self.selected_indices_: np.ndarray | None = None
        self.selected_ranked_: List[tuple[int, float, str]] | None = None
        self.view_composition_: Dict[str, int] = {}

    @staticmethod
    def _detect_knee_point(scores_sorted_desc: np.ndarray) -> int:
        if len(scores_sorted_desc) < 2:
            return 1
        y = scores_sorted_desc.astype(np.float32)
        x = np.arange(len(y), dtype=np.float32)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
        p1 = np.array([0.0, y_norm[0]], dtype=np.float32)
        p2 = np.array([1.0, y_norm[-1]], dtype=np.float32)
        vec_line = p2 - p1
        vec_point = np.vstack([x_norm, y_norm]).T - p1
        dist = np.abs(np.cross(vec_line, vec_point)) / (np.linalg.norm(vec_line) + 1e-9)
        return int(np.argmax(dist)) + 1

    def _score_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.quick_metric == "macro_f1":
            return float(f1_score(y_true, y_pred, average="macro"))
        return float(accuracy_score(y_true, y_pred))

    def _pick_k_by_valid(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        sorted_idx: np.ndarray,
        scores_sorted: np.ndarray,
    ) -> int:
        local_min_k = max(1, min(self.min_k, X_tr.shape[1]))
        local_max_k = max(local_min_k, min(self.max_k, X_tr.shape[1]))
        knee = self._detect_knee_point(scores_sorted)
        cand = set(self.k_candidates)
        cand.update({knee, knee * 2, knee * 3})
        cand = sorted(k for k in cand if local_min_k <= k <= local_max_k)
        if not cand:
            return local_max_k

        best_score = -1e9
        scores_map: Dict[int, float] = {}
        for k in cand:
            feat = sorted_idx[:k]
            Xtr_k = X_tr[:, feat]
            Xva_k = X_va[:, feat]
            Xtr_s, stats = standardize_fit_transform(Xtr_k)
            Xva_s = standardize_transform(Xva_k, stats)
            pred = CentroidVerifier().fit(Xtr_s, y_tr).predict(Xva_s)
            score = self._score_metric(y_va, pred)
            scores_map[k] = score
            best_score = max(best_score, score)

        threshold = best_score - self.within_delta
        for k in cand:
            if scores_map[k] >= threshold:
                return k
        return max(cand, key=lambda kk: scores_map[kk])

    def fit(
        self,
        X_list: List[np.ndarray],
        y: np.ndarray,
        view_global_indices: List[List[int]],
        X_valid_list: List[np.ndarray] | None = None,
        y_valid: np.ndarray | None = None,
        view_names: List[str] | None = None,
    ):
        if view_names is None:
            view_names = [f"View_{i}" for i in range(len(X_list))]

        use_valid = (
            self.k_mode == "valid"
            and X_valid_list is not None
            and y_valid is not None
            and len(X_valid_list) == len(X_list)
        )
        scores_list = self.scorer.fit_score_multi(X_list)
        selected_ranked: List[tuple[int, float, str]] = []
        self.view_composition_ = {}

        for v, X_view in enumerate(X_list):
            if X_view.shape[1] == 0:
                continue
            scores = np.asarray(scores_list[v], dtype=np.float32)
            sorted_idx = np.argsort(scores)[::-1]
            scores_sorted = scores[sorted_idx]
            local_min_k = max(1, min(self.min_k, X_view.shape[1]))
            local_max_k = max(local_min_k, min(self.max_k, X_view.shape[1]))
            final_k = max(local_min_k, min(self._detect_knee_point(scores_sorted), local_max_k))

            if use_valid:
                final_k = self._pick_k_by_valid(X_view, y, X_valid_list[v], y_valid, sorted_idx, scores_sorted)
                final_k = max(local_min_k, min(final_k, local_max_k))

            gidx = np.asarray(view_global_indices[v], dtype=np.int32)
            for local_i in sorted_idx[:final_k]:
                selected_ranked.append((int(gidx[local_i]), float(scores[local_i]), str(view_names[v])))
            self.view_composition_[view_names[v]] = int(final_k)

        selected_ranked.sort(key=lambda x: x[1], reverse=True)
        self.selected_ranked_ = selected_ranked
        self.selected_indices_ = np.array([x[0] for x in selected_ranked], dtype=np.int32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit before transform.")
        return X[:, self.selected_indices_]


def print_view_composition(stage_name: str, composition_dict: Dict[str, int], total_features: int) -> None:
    print("-" * 60)
    print(f"[{stage_name}] selected={total_features}")
    for view_name, count in sorted(composition_dict.items(), key=lambda x: x[1], reverse=True):
        ratio = 100.0 * count / max(1, total_features)
        print(f"  {view_name:<12}: {count:>4} ({ratio:>5.1f}%)")
    print("-" * 60)
