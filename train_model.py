"""
Train main model: Dual-Decoder HybridVAE
- VAE encoder on user interaction vector x_u
- Decoder = alpha * collaborative head (trainable item factors) +
            (1-alpha) * frozen text-embedding head (Sentence-BERT / LM embeddings)

Inspired by "frozen embedding decoder" HybridVAE, BUT different:
we add a learned collaborative decoder head + fusion (dual decoder), more robust on MovieLens.

Outputs:
- artifacts/item_embeddings.npy + artifacts/item_id_map.json (+meta)
- artifacts/DualHybridVAE.pt (model weights)
- results/metrics/DualHybridVAE_results.json (for evaluate.py)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, Precision
from cornac.models.recommender import Recommender

from src.utils.data_loader import load_movielens_100k
from src.utils.evaluation import save_results
from src.config import RANDOM_SEED, EMBEDDING_DIM, N_EPOCHS, METRICS_DIR


# -------------------------
# Hyperparams (env-tunable)
# -------------------------
LATENT_DIM = int(os.environ.get("VAE_LATENT_DIM", str(EMBEDDING_DIM)))
HIDDEN_DIMS = os.environ.get("VAE_HIDDEN_DIMS", "600,200")
DROPOUT = float(os.environ.get("VAE_DROPOUT", "0.5"))
BETA_MAX = float(os.environ.get("VAE_BETA", "0.2"))
ANNEAL_STEPS = int(os.environ.get("VAE_ANNEAL_STEPS", "2000"))
BATCH_SIZE = int(os.environ.get("VAE_BATCH_SIZE", "256"))
LR = float(os.environ.get("VAE_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("VAE_WEIGHT_DECAY", "0.0"))
ALPHA = float(os.environ.get("VAE_ALPHA", "0.8"))

TEXT_EMB_MODEL = os.environ.get("TEXT_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ARTIFACTS_DIR = project_root / "artifacts"
EMB_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
MAP_PATH = ARTIFACTS_DIR / "item_id_map.json"
META_PATH = ARTIFACTS_DIR / "checkpoint_meta.json"
CKPT_PATH = ARTIFACTS_DIR / "DualHybridVAE.pt"


# -------------------------
# MovieLens item text
# -------------------------
def _try_read_u_item() -> Optional[Path]:
    candidates = [
        project_root / "data" / "ml-100k" / "u.item",
        project_root / "data" / "raw" / "ml-100k" / "u.item",
        project_root / "data" / "MovieLens" / "ml-100k" / "u.item",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_item_text_map(train_set) -> Dict[str, str]:
    """raw_item_id -> text (title + genres). Falls back if u.item missing."""
    u_item = _try_read_u_item()
    item_text_from_file: Dict[str, str] = {}

    if u_item is not None:
        genre_names = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        with open(u_item, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 6:
                    continue
                iid = parts[0]
                title = parts[1]
                flags = parts[5:5 + len(genre_names)]
                genres = [g for g, fl in zip(genre_names, flags) if fl == "1" and g != "unknown"]
                item_text_from_file[iid] = title + (" | " + " ".join(genres) if genres else "")

    out: Dict[str, str] = {}
    for raw_iid in train_set.iid_map.keys():
        out[raw_iid] = item_text_from_file.get(raw_iid, f"Movie item {raw_iid}")
    return out


# -------------------------
# Text embeddings checkpoint
# -------------------------
def compute_text_embeddings(texts: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sentence-transformers.\n"
            "Install it with: pip install sentence-transformers\n"
            f"Original error: {repr(e)}"
        )

    model = SentenceTransformer(TEXT_EMB_MODEL)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    return emb.astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def load_or_make_item_embeddings(train_set) -> Tuple[np.ndarray, Dict[str, int]]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if EMB_PATH.exists() and MAP_PATH.exists():
        item_emb = np.load(EMB_PATH).astype(np.float32)
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            raw_item_to_row = json.load(f)
        return item_emb, {k: int(v) for k, v in raw_item_to_row.items()}

    item_text = build_item_text_map(train_set)
    raw_items = sorted(item_text.keys(), key=lambda x: train_set.iid_map[x])
    texts = [item_text[i] for i in raw_items]

    print(f"[embeddings] Computing embeddings for {len(texts)} items using: {TEXT_EMB_MODEL}")
    item_emb = compute_text_embeddings(texts)

    raw_item_to_row = {iid: idx for idx, iid in enumerate(raw_items)}
    np.save(EMB_PATH, item_emb)
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_item_to_row, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"text_model": TEXT_EMB_MODEL, "n_items": int(item_emb.shape[0]), "d": int(item_emb.shape[1]), "seed": int(RANDOM_SEED)},
            f,
            indent=2,
        )

    print(f"[embeddings] Saved checkpoint to {EMB_PATH} and {MAP_PATH}")
    return item_emb.astype(np.float32), raw_item_to_row


# -------------------------
# Dual-Decoder VAE (PyTorch)
# -------------------------
def _parse_hidden_dims(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class DualHybridVAE(Recommender):
    """
    Cornac-compatible wrapper around a PyTorch VAE.
    score(u) returns scores for all items.
    """

    def __init__(
        self,
        E_internal: np.ndarray,
        alpha: float = 0.8,
        latent_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.5,
        beta_max: float = 0.2,
        anneal_steps: int = 2000,
        lr: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 30,
        weight_decay: float = 0.0,
        seed: int = 42,
        name: str = "DualHybridVAE",
        verbose: bool = True,
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.E = E_internal.astype(np.float32)
        self.alpha = float(alpha)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = hidden_dims or [600, 200]
        self.dropout = float(dropout)
        self.beta_max = float(beta_max)
        self.anneal_steps = int(anneal_steps)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)

        self._torch = None
        self._model = None
        self._X_train = None   # torch tensor (U, I)
        self._E_t = None       # cached torch tensor (I, d) on correct device

    def _build_torch_model(self, n_items: int, d_text: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        hidden = self.hidden_dims
        latent_dim = self.latent_dim  # closure

        def mlp(in_dim: int, layers: List[int]) -> nn.Sequential:
            mods = []
            prev = in_dim
            for h in layers:
                mods.append(nn.Linear(prev, h))
                mods.append(nn.Tanh())
                prev = h
            return nn.Sequential(*mods)

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = mlp(n_items, hidden)
                last_dim = hidden[-1] if hidden else n_items

                self.mu = nn.Linear(last_dim, latent_dim)
                self.logvar = nn.Linear(last_dim, latent_dim)

                # collaborative head
                self.Q = nn.Parameter(0.01 * torch.randn(n_items, latent_dim))

                # text head: z -> d_text
                self.W_text = nn.Linear(latent_dim, d_text, bias=False)

                # item bias
                self.b = nn.Parameter(torch.zeros(n_items))

            def encode(self, x):
                h = self.enc(x) if len(hidden) > 0 else x
                return self.mu(h), self.logvar(h)

            def reparam(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x, E_frozen, alpha):
                mu, logvar = self.encode(x)
                z = self.reparam(mu, logvar)

                logits_cf = z @ self.Q.t()
                z_text = self.W_text(z)
                logits_text = z_text @ E_frozen.t()

                logits = alpha * logits_cf + (1.0 - alpha) * logits_text + self.b
                return logits, mu, logvar

            def score_from_mu(self, x, E_frozen, alpha):
                mu, _ = self.encode(x)
                logits_cf = mu @ self.Q.t()
                z_text = self.W_text(mu)
                logits_text = z_text @ E_frozen.t()
                return alpha * logits_cf + (1.0 - alpha) * logits_text + self.b

        return VAE()

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        try:
            import torch
            import torch.nn.functional as F
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: torch.\n"
                "Install it with: pip install torch\n"
                f"Original error: {repr(e)}"
            )

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        U = train_set.num_users
        I = train_set.num_items
        d_text = self.E.shape[1]

        # Dense implicit X (U x I)
        X = np.zeros((U, I), dtype=np.float32)
        u_idx, i_idx, _ = train_set.uir_tuple
        X[u_idx, i_idx] = 1.0

        device = torch.device("cpu")
        X_t = torch.from_numpy(X).to(device)
        E_t = torch.from_numpy(self.E).to(device)

        self._X_train = X_t
        self._E_t = E_t

        self._torch = torch
        self._model = self._build_torch_model(I, d_text).to(device)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        global_step = 0
        n_batches = int(np.ceil(U / self.batch_size))

        if self.verbose:
            print(f"[{self.name}] users={U}, items={I}, d_text={d_text}")
            print(f"[{self.name}] hidden={self.hidden_dims}, latent={self.latent_dim}, alpha={self.alpha}")
            print(f"[{self.name}] epochs={self.n_epochs}, batch={self.batch_size}, lr={self.lr}, beta_max={self.beta_max}")

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            perm = rng.permutation(U)

            self._model.train()
            epoch_loss = epoch_recon = epoch_kl = 0.0

            for bi in range(n_batches):
                idx = perm[bi * self.batch_size: (bi + 1) * self.batch_size]
                xb = X_t[idx]

                # input dropout (denoising)
                if self.dropout > 0:
                    mask = (torch.rand_like(xb) > self.dropout).float()
                    x_in = xb * mask
                else:
                    x_in = xb

                logits, mu, logvar = self._model(x_in, E_t, self.alpha)

                log_probs = F.log_softmax(logits, dim=1)
                recon = -(xb * log_probs).sum(dim=1).mean()

                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

                beta = min(self.beta_max, self.beta_max * (global_step / self.anneal_steps)) if self.anneal_steps > 0 else self.beta_max
                loss = recon + beta * kl

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss.detach().cpu())
                epoch_recon += float(recon.detach().cpu())
                epoch_kl += float(kl.detach().cpu())
                global_step += 1

            dt = time.time() - t0
            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.n_epochs):
                print(f"[{self.name}] epoch {epoch:03d}/{self.n_epochs} "
                      f"loss={epoch_loss/n_batches:.4f} recon={epoch_recon/n_batches:.4f} kl={epoch_kl/n_batches:.4f} "
                      f"time={dt:.2f}s")

        # Save checkpoint
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": {
                    "alpha": self.alpha,
                    "latent_dim": self.latent_dim,
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                    "beta_max": self.beta_max,
                    "anneal_steps": self.anneal_steps,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "seed": self.seed,
                },
            },
            CKPT_PATH,
        )
        if self.verbose:
            print(f"[{self.name}] Saved model checkpoint to: {CKPT_PATH}")

        return self

    def score(self, user_idx, item_idx=None):
        if self._model is None or self._X_train is None or self._E_t is None:
            raise RuntimeError("Model not fitted.")

        torch = self._torch
        self._model.eval()

        with torch.no_grad():
            x = self._X_train[user_idx].unsqueeze(0)  # (1, I)
            logits = self._model.score_from_mu(x, self._E_t, self.alpha).squeeze(0)  # (I,)

            if item_idx is None:
                return logits.cpu().numpy()
            return float(logits[item_idx].cpu().item())


# -------------------------
# Train + evaluate via Cornac
# -------------------------
def train_main_model():
    print("\n" + "=" * 60)
    print("Training Main Model: DualHybridVAE (dual decoder + frozen text embeddings)")
    print("=" * 60 + "\n")

    data = load_movielens_100k()

    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        rating_threshold=0.0,
        seed=RANDOM_SEED,
        exclude_unknowns=True,
        verbose=True
    )

    # Embeddings checkpoint (raw order)
    item_emb_raw, raw_item_to_row = load_or_make_item_embeddings(ratio_split.train_set)

    # Align embeddings to internal item ids
    I = ratio_split.train_set.num_items
    d = item_emb_raw.shape[1]
    E_internal = np.zeros((I, d), dtype=np.float32)
    for raw_iid, internal_iid in ratio_split.train_set.iid_map.items():
        row = raw_item_to_row.get(raw_iid, None)
        if row is not None:
            E_internal[internal_iid] = item_emb_raw[row]
    E_internal = normalize_rows(E_internal)

    hidden_dims = _parse_hidden_dims(HIDDEN_DIMS)
    model = DualHybridVAE(
        E_internal=E_internal,
        alpha=ALPHA,
        latent_dim=LATENT_DIM,
        hidden_dims=hidden_dims,
        dropout=DROPOUT,
        beta_max=BETA_MAX,
        anneal_steps=ANNEAL_STEPS,
        lr=LR,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        seed=RANDOM_SEED,
        name="DualHybridVAE",
        verbose=True
    )

    metrics = [
        Recall(k=5), Recall(k=10), Recall(k=20),
        NDCG(k=5), NDCG(k=10), NDCG(k=20),
        Precision(k=5), Precision(k=10), Precision(k=20),
    ]

    print("\nTraining and evaluating model...")
    start = time.time()

    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[model],
        metrics=metrics,
        user_based=True
    )
    exp.run()

    training_time = time.time() - start
    print(f"\nCompleted in {training_time:.2f} seconds")

    res = exp.result[0]
    results = {
        "model": "DualHybridVAE",
        "training_time": float(training_time),
        "alpha": float(ALPHA),
        "latent_dim": int(LATENT_DIM),
        "hidden_dims": hidden_dims,
        "dropout": float(DROPOUT),
        "beta_max": float(BETA_MAX),
        "anneal_steps": int(ANNEAL_STEPS),
        "lr": float(LR),
        "batch_size": int(BATCH_SIZE),
        "checkpoint": {
            "item_embeddings": str(EMB_PATH.relative_to(project_root)) if EMB_PATH.exists() else None,
            "item_id_map": str(MAP_PATH.relative_to(project_root)) if MAP_PATH.exists() else None,
            "model": str(CKPT_PATH.relative_to(project_root)) if CKPT_PATH.exists() else None,
        }
    }

    if hasattr(res, "metric_avg_results"):
        for metric_name, value in res.metric_avg_results.items():
            try:
                results[metric_name] = float(value)
            except Exception:
                results[metric_name] = value

    print("\nResults (TEST):")
    if "Recall@10" in results:
        print(f"  Recall@10:    {results['Recall@10']:.4f}")
    if "NDCG@10" in results:
        print(f"  NDCG@10:      {results['NDCG@10']:.4f}")
    if "Precision@10" in results:
        print(f"  Precision@10: {results['Precision@10']:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = METRICS_DIR / "DualHybridVAE_results.json"
    save_results(results, str(save_path))
    print(f"\nSaved metrics to: {save_path}")

    return model, results


if __name__ == "__main__":
    train_main_model()
