"""Bradley-Terry / RankNet surrogate in torch, trained on the comparison log.

A shared feature encoder (k-mer counts and/or latent coordinates) maps a sequence to
a scalar strength ``f(seq)``. The probability that A beats B is ``sigmoid(f(A)-f(B))``,
trained with logistic loss on the recorded verdicts. This lets us pre-rank many
candidates cheaply between oracle calls and refresh hourly during the event via
:meth:`BradleyTerrySurrogate.retrain_from_cache`.

To avoid leakage the train/val split is **by sequence, not by pair**: a validation
pair is one whose *both* sequences are held out, so the reported pairwise accuracy
reflects generalization to unseen sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import seqs
from .interfaces import Comparison, Navigator


@dataclass(slots=True)
class SurrogateConfig:
    """Configuration for the surrogate's features and training."""

    use_kmer: bool = True
    use_latent: bool = False
    ks: tuple[int, ...] = (4,)
    hidden: tuple[int, ...] = (64, 32)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 20
    val_frac: float = 0.2
    seed: int = 0


class _MLP(nn.Module):
    """Feed-forward encoder producing a scalar strength."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar strength for each row of ``x``."""
        return self.net(x).squeeze(-1)


@dataclass(slots=True)
class TrainReport:
    """Outcome of a training run."""

    val_accuracy: float
    best_epoch: int
    n_train_pairs: int
    n_val_pairs: int
    history: list[float] = field(default_factory=list)


class BradleyTerrySurrogate:
    """Torch Bradley-Terry model over sequence features."""

    def __init__(
        self,
        config: SurrogateConfig | None = None,
        navigator: Navigator | None = None,
    ) -> None:
        """Initialize the surrogate.

        Args:
            config: Feature/training configuration.
            navigator: Required only when ``config.use_latent`` is true.
        """
        self.cfg = config or SurrogateConfig()
        if self.cfg.use_latent and navigator is None:
            raise ValueError("use_latent=True requires a navigator")
        self.nav = navigator
        self.model: _MLP | None = None
        self._feat_cache: dict[str, np.ndarray] = {}
        self._in_dim: int | None = None

    def _features(self, seq: str) -> np.ndarray:
        """Return the (cached) feature vector for ``seq``."""
        cached = self._feat_cache.get(seq)
        if cached is not None:
            return cached
        parts: list[np.ndarray] = []
        if self.cfg.use_kmer:
            parts.append(seqs.kmer_vector_multi(seq, self.cfg.ks))
        if self.cfg.use_latent:
            assert self.nav is not None
            parts.append(np.asarray(self.nav.encode(seq), dtype=np.float32))
        vec = np.concatenate(parts).astype(np.float32)
        self._feat_cache[seq] = vec
        return vec

    def _split_by_sequence(
        self, comparisons: list[Comparison]
    ) -> tuple[list[Comparison], list[Comparison]]:
        """Split comparisons so val pairs have both sequences held out."""
        rng = np.random.default_rng(self.cfg.seed)
        unique = sorted({s for c in comparisons for s in (c.seq_a, c.seq_b)})
        rng.shuffle(unique)
        n_val = int(len(unique) * self.cfg.val_frac)
        val_seqs = set(unique[:n_val])
        train: list[Comparison] = []
        val: list[Comparison] = []
        for c in comparisons:
            a_val = c.seq_a in val_seqs
            b_val = c.seq_b in val_seqs
            if a_val and b_val:
                val.append(c)
            elif not a_val and not b_val:
                train.append(c)
            # Cross pairs (one side held out) are dropped to prevent leakage.
        return train, val

    def _tensors(self, comps: list[Comparison]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorize comparisons into (features_a, features_b, label) tensors."""
        fa = np.stack([self._features(c.seq_a) for c in comps])
        fb = np.stack([self._features(c.seq_b) for c in comps])
        y = np.array([1.0 if c.winner == "A" else 0.0 for c in comps], dtype=np.float32)
        return (
            torch.from_numpy(fa),
            torch.from_numpy(fb),
            torch.from_numpy(y),
        )

    def fit(self, comparisons: list[Comparison]) -> TrainReport:
        """Train with early stopping; return held-out pairwise accuracy.

        Args:
            comparisons: The full comparison log.

        Returns:
            A :class:`TrainReport`.
        """
        if len(comparisons) < 4:
            raise ValueError("need at least 4 comparisons to train")
        torch.manual_seed(self.cfg.seed)
        train, val = self._split_by_sequence(comparisons)
        if not train:
            raise ValueError("no training pairs after by-sequence split; add more data")
        xa, xb, y = self._tensors(train)
        self._in_dim = xa.shape[1]
        self.model = _MLP(self._in_dim, self.cfg.hidden)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        val_tensors = self._tensors(val) if val else None
        best_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        wait = 0
        history: list[float] = []

        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            opt.zero_grad()
            logits = self.model(xa) - self.model(xb)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            acc = self._accuracy(val_tensors) if val_tensors else self._accuracy((xa, xb, y))
            history.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainReport(
            val_accuracy=best_acc,
            best_epoch=best_epoch,
            n_train_pairs=len(train),
            n_val_pairs=len(val),
            history=history,
        )

    @torch.no_grad()
    def _accuracy(self, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """Pairwise accuracy on the given (fa, fb, label) tensors."""
        assert self.model is not None
        xa, xb, y = tensors
        self.model.eval()
        pred = (self.model(xa) - self.model(xb) > 0).float()
        return float((pred == y).float().mean().item())

    @torch.no_grad()
    def score(self, seq: str) -> float:
        """Return the scalar strength ``f(seq)`` (higher is stronger)."""
        if self.model is None:
            raise RuntimeError("surrogate is not trained")
        self.model.eval()
        x = torch.from_numpy(self._features(seq)).unsqueeze(0)
        return float(self.model(x).item())

    def rank(self, sequences: list[str]) -> list[str]:
        """Return sequences ordered strongest-first by surrogate score."""
        return sorted(sequences, key=self.score, reverse=True)

    def retrain_from_cache(self, path: str | Path) -> TrainReport:
        """Reload the JSONL comparison cache and refit (hourly-refresh convenience)."""
        from .cache import ComparisonCache

        cache = ComparisonCache(path)
        comps = list(cache._store.values())
        cache.close()
        return self.fit(comps)
