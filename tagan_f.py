from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class Generator(nn.Module):
    def __init__(self, n_classes: int, input_dim: int, noise_dim: int = 128, emb_dim: Optional[int] = None):
        super().__init__()
        if emb_dim is None:
            emb_dim = min(64, max(16, n_classes // 2))
        self.noise_dim = int(noise_dim)
        self.label_emb = nn.Embedding(int(n_classes), int(emb_dim))
        self.model = nn.Sequential(
            nn.Linear(self.noise_dim + emb_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        return self.model(torch.cat([noise, emb], dim=1))


class Discriminator(nn.Module):
    def __init__(self, n_classes: int, input_dim: int, emb_dim: Optional[int] = None):
        super().__init__()
        if emb_dim is None:
            emb_dim = min(64, max(16, n_classes // 2))
        self.label_emb = nn.Embedding(int(n_classes), int(emb_dim))
        self.model = nn.Sequential(
            nn.Linear(input_dim + emb_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        return self.model(torch.cat([x, emb], dim=1))


class ProxyClassifier(nn.Module):
    def __init__(self, n_classes: int, input_dim: int, feat_dim: int = 256):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        h = self.feat(x)
        logits = self.head(h)
        if return_feat:
            return logits, h
        return logits


class FeatureTAGAN_Feedback:
    def __init__(
        self,
        n_classes: int,
        input_dim: int,
        device: str = "cuda",
        lr: float = 1e-4,
        noise_dim: int = 128,
        standardize: bool = True,
        lambda_gp: float = 10.0,
        lambda_cls: float = 0.5,
        lambda_sem: float = 0.10,
        lambda_corr: float = 0.02,
        lambda_fm: float = 0.05,
        max_corr_dim: int = 128,
        feature_names: Optional[List[str]] = None,
        feat_names: Optional[List[str]] = None,
        **kwargs,
    ):
        if kwargs:
            print(f"[TAGAN-F] ignored init kwargs: {list(kwargs.keys())[:8]}")
        self.n_classes = int(n_classes)
        self.input_dim = int(input_dim)
        self.device = str(device)
        self.noise_dim = int(noise_dim)
        self.standardize = bool(standardize)
        self.lambda_gp = float(lambda_gp)
        self.lambda_cls = float(lambda_cls)
        self.lambda_sem = float(lambda_sem)
        self.lambda_corr = float(lambda_corr)
        self.lambda_fm = float(lambda_fm)
        self.max_corr_dim = int(max_corr_dim)
        self.feature_names = list(feat_names) if feat_names is not None else (list(feature_names) if feature_names is not None else None)

        self.G = Generator(self.n_classes, self.input_dim, noise_dim=self.noise_dim).to(self.device)
        self.D = Discriminator(self.n_classes, self.input_dim).to(self.device)
        self.C = ProxyClassifier(self.n_classes, self.input_dim).to(self.device)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr=float(lr), betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=float(lr), betas=(0.0, 0.9))
        self.optimizer_C = optim.Adam(self.C.parameters(), lr=float(lr), betas=(0.0, 0.9))
        self.criterion_cls = nn.CrossEntropyLoss()

        self._mu: Optional[torch.Tensor] = None
        self._sigma: Optional[torch.Tensor] = None
        self._xmin: Optional[torch.Tensor] = None
        self._xmax: Optional[torch.Tensor] = None
        self._nonneg_idx: Optional[torch.Tensor] = None
        self._triples: List[Tuple[int, int, int]] = []
        self._corr_feat_idx: Optional[torch.Tensor] = None
        self._corr_global: Optional[torch.Tensor] = None
        self._corr_by_class: List[Optional[torch.Tensor]] = []

    def _fit_scaler_np(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self._mu = torch.tensor(X.mean(axis=0), device=self.device, dtype=torch.float32)
        self._sigma = torch.tensor(X.std(axis=0) + 1e-6, device=self.device, dtype=torch.float32)
        self._xmin = torch.tensor(X.min(axis=0), device=self.device, dtype=torch.float32)
        self._xmax = torch.tensor(X.max(axis=0), device=self.device, dtype=torch.float32)

    def _preprocess_np(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.standardize:
            return X
        mu = self._mu.detach().cpu().numpy()
        sigma = self._sigma.detach().cpu().numpy()
        return ((X - mu) / sigma).astype(np.float32, copy=False)

    def _postprocess_torch(self, X_std: torch.Tensor, clip: bool = True) -> torch.Tensor:
        out = X_std if not self.standardize else (X_std * self._sigma + self._mu)
        if clip:
            out = torch.max(torch.min(out, self._xmax), self._xmin)
        return out

    def _build_semantic_constraints(self, X_raw: np.ndarray, feature_names: Optional[List[str]]) -> None:
        X_raw = np.asarray(X_raw, dtype=np.float32)
        neg_ratio = np.mean(X_raw < 0.0, axis=0)
        nonneg_idx = np.where(neg_ratio < 0.01)[0]
        self._nonneg_idx = torch.tensor(nonneg_idx, device=self.device, dtype=torch.long) if len(nonneg_idx) else None

        self._triples = []
        if not feature_names:
            return
        name_to_idx = {str(name).lower(): i for i, name in enumerate(feature_names)}
        for name, idx_mean in name_to_idx.items():
            if not name.endswith("_mean"):
                continue
            prefix = name[:-5]
            idx_min = name_to_idx.get(prefix + "_min")
            idx_max = name_to_idx.get(prefix + "_max")
            if idx_min is not None and idx_max is not None:
                self._triples.append((idx_min, idx_mean, idx_max))

    def _corr_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        cov = x.T @ x / max(1, x.shape[0] - 1)
        std = torch.sqrt(torch.diag(cov) + 1e-6)
        return cov / (std[:, None] * std[None, :] + 1e-6)

    def _prepare_corr_targets(self, X_std: np.ndarray, y_train: np.ndarray) -> None:
        take = min(X_std.shape[1], self.max_corr_dim)
        self._corr_feat_idx = torch.arange(take, device=self.device, dtype=torch.long)
        Xt = torch.tensor(X_std[:, :take], device=self.device, dtype=torch.float32)
        self._corr_global = self._corr_matrix(Xt).detach()

        y_train = np.asarray(y_train, dtype=np.int64)
        self._corr_by_class = []
        for class_idx in range(self.n_classes):
            mask = y_train == class_idx
            if int(np.sum(mask)) < 8:
                self._corr_by_class.append(None)
                continue
            Xc = torch.tensor(X_std[mask, :take], device=self.device, dtype=torch.float32)
            self._corr_by_class.append(self._corr_matrix(Xc).detach())

    def _semantic_loss(self, x_raw: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        if self._nonneg_idx is not None and len(self._nonneg_idx) > 0:
            loss = loss + torch.relu(-x_raw[:, self._nonneg_idx]).mean()
        for idx_min, idx_mean, idx_max in self._triples:
            vmin = x_raw[:, idx_min]
            vmean = x_raw[:, idx_mean]
            vmax = x_raw[:, idx_max]
            loss = loss + torch.relu(vmin - vmean).mean()
            loss = loss + torch.relu(vmean - vmax).mean()
        return loss

    def _corr_loss(self, x_std: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._corr_global is None or self._corr_feat_idx is None:
            return torch.zeros(1, device=self.device, dtype=torch.float32)
        losses = []
        for class_idx in torch.unique(labels).detach().cpu().tolist():
            class_idx = int(class_idx)
            mask = labels == class_idx
            if int(mask.sum().item()) < 8:
                continue
            target = None
            if 0 <= class_idx < len(self._corr_by_class):
                target = self._corr_by_class[class_idx]
            if target is None:
                target = self._corr_global
            corr = self._corr_matrix(x_std[mask][:, self._corr_feat_idx])
            losses.append(torch.mean((corr - target) ** 2))
        if not losses:
            return torch.zeros(1, device=self.device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def _feature_matching_loss(self, real_x: torch.Tensor, fake_x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, real_feat = self.C(real_x, return_feat=True)
        _, fake_feat = self.C(fake_x, return_feat=True)
        losses = []
        for class_idx in torch.unique(labels).detach().cpu().tolist():
            mask = labels == int(class_idx)
            if int(mask.sum().item()) < 2:
                continue
            real_c = real_feat[mask]
            fake_c = fake_feat[mask]
            mean_loss = torch.mean((real_c.mean(dim=0) - fake_c.mean(dim=0)) ** 2)
            std_loss = torch.mean((real_c.std(dim=0, unbiased=False) - fake_c.std(dim=0, unbiased=False)) ** 2)
            losses.append(mean_loss + std_loss)
        if not losses:
            return torch.zeros(1, device=self.device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def _gradient_penalty(self, real_x: torch.Tensor, fake_x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real_x.size(0), 1, device=self.device)
        interpolates = alpha * real_x + (1.0 - alpha) * fake_x
        interpolates.requires_grad_(True)
        d_inter = self.D(interpolates, labels)
        grad = autograd.grad(
            outputs=d_inter,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        lambda_gp: Optional[float] = None,
        n_critic: int = 5,
        print_every: int = 25,
        lambda_cls: Optional[float] = None,
        lambda_sem: Optional[float] = None,
        lambda_corr: Optional[float] = None,
        lambda_fm: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        feat_names: Optional[List[str]] = None,
    ):
        self.G.train()
        self.D.train()
        self.C.train()

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        if lambda_gp is not None:
            self.lambda_gp = float(lambda_gp)
        if lambda_cls is not None:
            self.lambda_cls = float(lambda_cls)
        if lambda_sem is not None:
            self.lambda_sem = float(lambda_sem)
        if lambda_corr is not None:
            self.lambda_corr = float(lambda_corr)
        if lambda_fm is not None:
            self.lambda_fm = float(lambda_fm)
        if feat_names is not None:
            self.feature_names = list(feat_names)
        elif feature_names is not None:
            self.feature_names = list(feature_names)

        self._fit_scaler_np(X_train)
        X_std = self._preprocess_np(X_train)
        if self.feature_names is not None and len(self.feature_names) != X_train.shape[1]:
            self.feature_names = None
        self._build_semantic_constraints(X_train, self.feature_names)
        self._prepare_corr_targets(X_std, y_train)

        x_tensor = torch.from_numpy(X_std).float()
        y_tensor = torch.from_numpy(y_train).long()
        counts = np.bincount(y_train, minlength=self.n_classes).astype(np.float32)
        class_weights = counts.sum() / np.maximum(counts, 1.0)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=int(batch_size), sampler=sampler, drop_last=True)

        for epoch in range(1, int(epochs) + 1):
            loss_d_last = 0.0
            loss_g_last = 0.0
            for real_x, labels in loader:
                real_x = real_x.to(self.device)
                labels = labels.to(self.device)

                for _ in range(int(n_critic)):
                    z = torch.randn(real_x.size(0), self.noise_dim, device=self.device)
                    fake_x = self.G(z, labels).detach()
                    d_real = self.D(real_x, labels).mean()
                    d_fake = self.D(fake_x, labels).mean()
                    gp = self._gradient_penalty(real_x, fake_x, labels)
                    loss_d = d_fake - d_real + self.lambda_gp * gp
                    self.optimizer_D.zero_grad()
                    loss_d.backward()
                    self.optimizer_D.step()
                    loss_d_last = float(loss_d.item())

                logits = self.C(real_x)
                loss_c = self.criterion_cls(logits, labels)
                self.optimizer_C.zero_grad()
                loss_c.backward()
                self.optimizer_C.step()

                z = torch.randn(real_x.size(0), self.noise_dim, device=self.device)
                fake_x = self.G(z, labels)
                adv = -self.D(fake_x, labels).mean()
                cls_loss = self.criterion_cls(self.C(fake_x), labels)
                fake_raw = self._postprocess_torch(fake_x, clip=False)
                sem_loss = self._semantic_loss(fake_raw)
                corr_loss = self._corr_loss(fake_x, labels)
                fm_loss = self._feature_matching_loss(real_x.detach(), fake_x, labels)
                loss_g = adv
                loss_g = loss_g + self.lambda_cls * cls_loss
                loss_g = loss_g + self.lambda_sem * sem_loss
                loss_g = loss_g + self.lambda_corr * corr_loss
                loss_g = loss_g + self.lambda_fm * fm_loss
                self.optimizer_G.zero_grad()
                loss_g.backward()
                self.optimizer_G.step()
                loss_g_last = float(loss_g.item())

            if epoch == 1 or epoch % int(print_every) == 0 or epoch == int(epochs):
                print(f"[TAGAN-F] epoch={epoch:>3d} loss_d={loss_d_last:.4f} loss_g={loss_g_last:.4f}")

    @torch.no_grad()
    def sample(self, n: int, class_idx: int, batch_gen: int = 2048, clip: bool = True) -> np.ndarray:
        self.G.eval()
        out = []
        done = 0
        while done < int(n):
            curr = min(int(batch_gen), int(n) - done)
            z = torch.randn(curr, self.noise_dim, device=self.device)
            labels = torch.full((curr,), int(class_idx), device=self.device, dtype=torch.long)
            gen = self.G(z, labels)
            gen = self._postprocess_torch(gen, clip=bool(clip))
            out.append(gen.detach().cpu().numpy())
            done += curr
        return np.vstack(out).astype(np.float32, copy=False)
