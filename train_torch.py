# train_torch.py — the ONE torch training loop (TSplineDCM + every BaseDCM baseline).
"""One-layer torch trainer on the fixed splits.

Replaces v1's two layers (experiments/torch_trainer.py wrapping tspline/tspline_trainer.py).
The loop is a faithful port of TSplineTrainer.train_split: same seeding order,
same grid-update schedule, same "mc" validation, same best-by-val-NLL checkpoint and early
stopping — a run reproduces the v1 numbers bit-for-bit on the same GPU.

    trainer = TorchDCMTrainer(dataset_name, model_class, model_config, dcm_config, train_config, ...)
    trainer.run(split_id, init_seed)          # BaseTrainer: load split -> _fit -> _predict_proba -> metrics.json
    trainer.prepare(seed); trainer.load_checkpoint(path)   # plots/extract: rebuild + load best.pt, no training

Run directory: {out_root}/{dataset}/{model_name}/split{nnn}_seed{s}/
    checkpoints/best.pt   model_state (+ optimizer/scheduler state) at the best validation epoch
    converge.csv          per-epoch train/val metrics (diagnostic only)
    metrics.json, predictions.npz   written by BaseTrainer
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from dcm_data import DcmConfig, DcmDataset
from metrics import probs_from_utilities
from models.config import EvalType, ModelConfig, TrainConfig, get_device
from trainer_base import BaseTrainer

PREDICT_CHUNK = 8192


# --------------------------------------------------------------------------- #
# loss + helpers (verbatim from v1 tspline_trainer.py)
# --------------------------------------------------------------------------- #
def masked_log_softmax(logits: Tensor, mask: Optional[Tensor]) -> Tensor:
    """log softmax over available alternatives only (-1e9 where mask == 0)."""
    if mask is None:
        return torch.log_softmax(logits, dim=-1)
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    return torch.log_softmax(masked_logits, dim=-1)


def weighted_f1(true: Tensor, pred: Tensor, n_classes: int) -> Tensor:
    """sklearn's f1_score(average="weighted") on the device: per-class 2tp/(2tp+fp+fn) (0 where undefined),
    weighted by the class's support in `true`. Float64 like sklearn; no host round trip."""
    cm = torch.bincount(true * n_classes + pred, minlength=n_classes * n_classes).view(n_classes, n_classes).double()
    tp = cm.diagonal()
    support = cm.sum(dim=1)            # true counts per class
    denom = 2.0 * tp + (cm.sum(dim=0) - tp) + (support - tp)
    f1c = torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1.0), torch.zeros_like(tp))
    return (f1c * support).sum() / support.sum().clamp_min(1.0)


def default_dcm_loss(
    model_output: Any, batch: dict[str, Tensor], device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    """NLL (+ the model's own weighted KLD term if it returns one) and batch metrics.

    model_output: {"V": [B, I] utilities, optional "P": [B, I] masked probabilities,
                   optional "kld": scalar}; batch["label"] is one-hot [B, I].
    The metrics are 0-d tensors on the device (no .item() / sklearn per step: those forced a GPU->CPU sync
    every step); the trainer reads them once per epoch.
    """
    V = model_output["V"].to(device)
    y = batch["label"].to(device)
    if "P" in model_output:
        logP = model_output["P"].to(device).clamp_min(1e-12).log()
    else:
        av = batch.get("av", None)
        av = av.to(device) if av is not None else None
        logP = masked_log_softmax(V, av)
    nll = -(y * logP).sum(dim=1).mean()
    kld = model_output["kld"].to(device) if "kld" in model_output else torch.tensor(0.0, device=device, dtype=nll.dtype)
    pen = model_output["penalty"].to(device) if "penalty" in model_output else None   # structural penalty (train mode)
    loss = nll + kld if pen is None else nll + kld + pen
    with torch.no_grad():
        pred = V.argmax(dim=1)
        true = y.argmax(dim=1)
        acc = (pred == true).float().mean()
        f1 = weighted_f1(true, pred, V.shape[1])
        brier = torch.mean(torch.sum((logP.exp() - y) ** 2, dim=1))
    return loss, {"loss": loss.detach(), "nll": nll.detach(), "kld": kld.detach(),
                  "penalty": pen.detach() if pen is not None else torch.zeros((), device=device, dtype=nll.dtype),
                  "acc": acc, "f1": f1, "brier": brier}


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _stdnorm_ppf(p: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    two = torch.tensor(2.0, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    return torch.sqrt(two) * torch.erfinv(two * torch.tensor(p, device=device, dtype=dtype) - one)


class _VectorizedLoader:
    """Drop-in replacement for DataLoader(Subset(ds, idx)): one vectorized slice per batch
    (~40x faster than per-row __getitem__). Shuffling draws from the global torch RNG, so
    runs stay seed-reproducible. `.dataset.indices` kept for code that wants the row ids."""

    def __init__(self, ds: DcmDataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> None:
        self._ds = ds
        self._indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.dataset = SimpleNamespace(indices=[int(i) for i in indices])

    def __len__(self) -> int:
        return math.ceil(len(self._indices) / self.batch_size)

    def __iter__(self) -> Iterator[dict]:
        idx = self._indices
        if self.shuffle:
            idx = idx[torch.randperm(len(idx))]
        for s in range(0, len(idx), self.batch_size):
            yield self._ds.make_batch_from_indices(idx[s: s + self.batch_size])


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class TorchDCMTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_name: str,
        model_class: type,
        model_config: ModelConfig,
        dcm_config: DcmConfig,
        train_config: TrainConfig,
        model_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        loss_fn: Optional[Callable] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(dataset_name=dataset_name, dcm_config=dcm_config,
                         model_name=model_name or model_class.__name__, **base_kwargs)
        self.model_class = model_class
        self.mcfg = model_config
        self.tcfg_template = train_config
        self.tcfg: TrainConfig = train_config
        self.model_kwargs: dict[str, Any] = dict(model_kwargs or {})  # e.g. {"cfg": benchmark_config}
        self.loss_fn = loss_fn or default_dcm_loss
        self.device: torch.device = get_device()
        # filled by prepare() / _fit()
        self.ds: DcmDataset | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.scaler: GradScaler | None = None

    # ---------- building blocks (also used by plots/extract and tests) ----------

    def dataset(self, device: torch.device | str | None = None, float_type: torch.dtype | None = None) -> DcmDataset:
        """Tensors of the whole csv on `device` (built once per trainer and reused across runs)."""
        dev = torch.device(device) if device is not None else self.device
        if self.ds is None or self.ds.device != dev or (float_type is not None and self.ds._float_type != float_type):
            self.ds = DcmDataset(config=self.dcfg, device=dev, float_type=float_type)
        return self.ds

    def build_model(self, seed: int) -> torch.nn.Module:
        """Seed everything, construct the model (+ optimizer/scheduler/scaler) — same order as v1."""
        _seed_everything(int(seed), deterministic=getattr(self.tcfg, "deterministic", False))
        ds = self.dataset()
        z_df = ds.data.iloc[:, self.dcfg.z_cols]  # full data decides the z grids (as v1)
        self.model = self.model_class(
            model_config=self.mcfg, dcm_config=self.dcfg, device=self.device, random_seed=int(seed),
            z_df=z_df, **self.model_kwargs,
        ).to(self.device)
        t = self.tcfg
        # fused_adam is opt-in (one kernel, other rounding). When off, do NOT pass fused=False: an explicit value makes
        # torch skip its foreach default and fall back to the single-tensor loop, which rounds differently from the
        # paper runs (their Adam was the foreach one) and is slower.
        fused_kw = {"fused": True} if bool(getattr(t, "fused_adam", False)) and self.device.type == "cuda" else {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(getattr(t, "lr", 1e-3)),
                                          weight_decay=float(getattr(t, "weight_decay", 0.0)),
                                          betas=tuple(getattr(t, "betas", (0.9, 0.999))), **fused_kw)
        sched = getattr(t, "scheduler", "none")
        if sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(getattr(t, "epochs", 50)))
        elif sched == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(getattr(t, "step_size", 20)),
                                                             gamma=float(getattr(t, "gamma", 0.5)))
        else:
            self.scheduler = None
        self.use_amp = bool(getattr(t, "amp", False))
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        if hasattr(self.model, "build_basis_cache") and bool(getattr(self.mcfg, "basis_cache", True)):
            self.model.build_basis_cache(ds)   # B-spline basis of every row, once; forward gathers by batch["idx"]
        return self.model

    def prepare(self, seed: int = 42, device: torch.device | str | None = None,
                float_type: torch.dtype | None = None) -> torch.nn.Module:
        """Dataset + untrained model, no training — for loading checkpoints in plots/extract and tests."""
        if device is not None:
            self.device = torch.device(device)
        self.dataset(self.device, float_type)
        return self.build_model(seed)

    def save_checkpoint(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        assert self.model is not None and self.optimizer is not None
        state: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "configs": {"ModelConfig": self.mcfg.__dict__, "DcmConfig": self.dcfg.__dict__, "TrainConfig": self.tcfg.__dict__},
        }
        if extra:
            state.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False, load_scheduler: bool = False) -> dict[str, Any]:
        assert self.model is not None, "call prepare()/build_model() first"
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"], strict=False, assign=True)  # allow missing keys (as v1)
        if load_optimizer and payload.get("optimizer_state") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if load_scheduler and payload.get("scheduler_state") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        return payload

    def load_trained(self, split_id: int, init_seed: int = 42) -> torch.nn.Module:
        """Rebuild the model and load this run's best.pt (no training)."""
        if self.model is None:
            self.prepare(init_seed)
        self.load_checkpoint(self.run_dir(split_id, init_seed) / "checkpoints" / "best.pt")
        self.model.eval()
        return self.model

    # ---------- one optimisation step / one evaluation pass ----------

    def step(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        assert self.model is not None and self.optimizer is not None and self.scaler is not None
        self.model.train()
        with torch.amp.autocast_mode.autocast(device_type=self.device.type, enabled=self.use_amp):
            output = self.model(batch)
            loss, metrics = self.loss_fn(output, batch, self.device)
        self.scaler.scale(loss).backward()
        grad_clip = float(getattr(self.tcfg, "grad_clip", 0.0))
        if grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.detach(), metrics

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Validation pass. eval_mode: "mu" (deterministic taste), "mc" (average over eval_mc_samples
        stochastic draws), "percentile" (fixed z-scores). Note: "nll" here is loss = nll + kld, exactly
        as v1 — it only drives model selection; metrics.json uses metrics.dcm_metrics on P."""
        assert self.model is not None
        t = self.tcfg
        mode: EvalType = getattr(t, "eval_mode", "mu")
        mc_samples = int(getattr(t, "eval_mc_samples", 1))
        percentiles = getattr(t, "eval_percentiles", None)
        sample_ratio = float(getattr(t, "eval_sample_ratio", 1.0))
        assert 0.0 < sample_ratio <= 1.0
        if mode == "percentile":
            ps = list(percentiles or [])
            assert ps and all(0.0 < p < 1.0 for p in ps), "percentiles must be in (0,1)"

        self.model.eval()
        totals = {"loss": 0.0, "acc": 0.0, "n": 0, "f1": 0.0, "brier": 0.0}

        def _one_pass(batch, sample_arg):
            output = self.model(batch, sample=sample_arg)
            loss, m = self.loss_fn(output, batch, self.device)
            return float(loss), float(m.get("acc", 0.0)), float(m.get("f1", 0.0)), float(m.get("brier", 0.0))

        def _acc(l, a, f, b, bs):
            totals["loss"] += l * bs
            totals["acc"] += a * bs
            totals["f1"] += f * bs
            totals["brier"] += b * bs
            totals["n"] += bs

        orig_seed = getattr(self.model, "random_seed", 0)
        for batch in loader:
            if random.random() > sample_ratio:
                continue
            bs = int(next(v for v in batch.values() if torch.is_tensor(v)).size(0))
            if mode == "mu":
                _acc(*_one_pass(batch, False), bs)
            elif mode == "mc":
                sums = [0.0, 0.0, 0.0, 0.0]
                for s in range(mc_samples):
                    mixed = (orig_seed ^ ((s + 1) * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF  # decorrelate draws
                    setattr(self.model, "random_seed", mixed)
                    for i, v in enumerate(_one_pass(batch, True)):
                        sums[i] += v
                setattr(self.model, "random_seed", orig_seed)
                _acc(*(v / mc_samples for v in sums), bs)
            elif mode == "percentile":
                dtype = next(self.model.parameters()).dtype
                zs = [_stdnorm_ppf(p, device=self.device, dtype=dtype).item() for p in ps]
                sums = [0.0, 0.0, 0.0, 0.0]
                for z in zs:
                    for i, v in enumerate(_one_pass(batch, float(z))):
                        sums[i] += v
                _acc(*(v / len(zs) for v in sums), bs)
            else:
                raise ValueError(f"Unknown eval_mode: {mode}")
        n = max(1, totals["n"])
        return {"nll": totals["loss"] / n, "acc": totals["acc"] / n, "f1": totals["f1"] / n, "brier": totals["brier"] / n}

    # ---------- grid update schedule (TSplineDCM only; baselines have no update_grid) ----------

    def _maybe_update_grid(self, epoch: int, train_idx: np.ndarray, gu: dict[str, Any]) -> None:
        assert self.model is not None and self.ds is not None
        if not gu["enabled"] or epoch > gu["epoch_cap"]:
            return
        pin_right = self.model.get_right_margin_from_full_ds(self.ds)  # margins from the full csv (as v1)
        clamp_left = self.model.get_left_margin_from_full_ds(self.ds)
        if getattr(self.mcfg, "MC_grid_fix_left_zero", False):
            clamp_left = {k: 0.0 for k in clamp_left}
        df = self.ds.data.loc[[int(i) for i in train_idx]]  # train rows only: no leakage
        if epoch == gu["warmup"] or (epoch >= gu["warmup"] and gu["every"] > 0 and epoch % gu["every"] == 0):
            self.model.update_grid(df, use_availability=True, sample_per_x=gu["sample_per_x"],
                                   clamp_left_tos=clamp_left, pin_right_core_tos=pin_right)

    # ---------- the loop ----------

    def _fit(self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
             init_seed: int, run_dir: Path) -> dict[str, Any]:
        t = dataclasses.replace(self.tcfg_template, random_seed=int(init_seed))
        self.tcfg = t
        ds = self.dataset()
        self.build_model(int(init_seed))
        model = self.model
        assert model is not None
        bs = int(t.batch_size)
        train_loader = _VectorizedLoader(ds, train_idx, bs, shuffle=True)
        val_loader = _VectorizedLoader(ds, val_idx, bs, shuffle=False)

        # grid update schedule (v1 semantics: GU_sample_per_x null -> use all rows AND only the one-shot warm-up update)
        m = self.mcfg
        gu_every = int(getattr(m, "GU_every", 0))
        sample_per_x = getattr(m, "GU_sample_per_x", None)
        if sample_per_x is None:
            sample_per_x, gu_every = len(ds), 0
        gu = {"enabled": int(getattr(m, "GU_every", 0)) > 0 and hasattr(model, "update_grid"), "every": gu_every,
              "warmup": int(getattr(m, "GU_warmup", 1)), "epoch_cap": int(getattr(m, "GU_epoch_cap", 100)),
              "sample_per_x": sample_per_x}

        epochs = int(getattr(t, "epochs", 50))
        eval_every = max(1, int(getattr(t, "eval_every", 1)))
        patience = getattr(t, "early_stopping_patience", None)
        compact = bool(getattr(t, "compact_log", False))
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"

        best_val_nll, best_info, bad_epochs = math.inf, {}, 0
        rows: list[dict[str, Any]] = []
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            if hasattr(model, "reset_eps_cache"):
                model.reset_eps_cache()
            self._maybe_update_grid(epoch, train_idx, gu)
            train_metrics: list[dict[str, float]] = []
            for batch in train_loader:
                _, mtr = self.step(batch)
                train_metrics.append(mtr)
            if self.scheduler is not None:
                self.scheduler.step()
            agg = {k: float(torch.stack([torch.as_tensor(x[k], dtype=torch.float64, device=self.device) for x in train_metrics]).nanmean().item())
                   for k in train_metrics[0]} if train_metrics else {}   # one host read per metric per epoch
            row: dict[str, Any] = {"epoch": epoch, **{f"train_{k}": v for k, v in agg.items()}}

            if epoch % eval_every == 0:
                val = self.evaluate(val_loader) if len(val_loader) > 0 else {}
                row.update({f"val_{k}": v for k, v in val.items()})
                if compact:
                    print(f"Epoch [{epoch:03d}/{epochs:03d}] | train: nll={agg.get('nll', float('nan')):.4f} "
                          f"acc={agg.get('acc', float('nan')):.4f} kld={agg.get('kld', float('nan')):.4f} | "
                          f"val({t.eval_mode}): nll={val.get('nll', float('nan')):.4f} acc={val.get('acc', float('nan')):.4f}")
                if "nll" in val and val["nll"] < best_val_nll:
                    bad_epochs = 0
                    best_val_nll = float(val["nll"])
                    best_info = {"epoch": epoch, "best_val_nll": best_val_nll, "val_acc": val["acc"]}
                    self.save_checkpoint(best_path, extra=best_info)
                elif not val:  # no validation set: run all epochs, keep the last state
                    bad_epochs = 0
                    self.save_checkpoint(best_path, extra={"epoch": epoch})
                else:
                    bad_epochs += 1
                if patience is not None and bad_epochs >= patience:
                    rows.append(row)
                    break
            rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "converge.csv", index=False)  # diagnostic only
        train_seconds = time.time() - t0
        print(f"[fit] {self.model_name} split done: best val nll {best_val_nll:.4f} @ epoch "
              f"{best_info.get('epoch', -1)} ({len(rows)} epochs, {train_seconds:.0f}s)")

        # serve predictions from the best-val checkpoint, not the last epoch
        best_extra: dict[str, Any] = {}
        if best_path.exists():
            payload = self.load_checkpoint(best_path)
            best_extra = {k: v for k, v in payload.items()
                          if k not in ("model_state", "optimizer_state", "scheduler_state", "configs")
                          and not (isinstance(v, float) and not math.isfinite(v))}
        model.eval()
        return {
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "best_checkpoint": best_extra,
            "trainer_best_val_nll": float(best_val_nll) if math.isfinite(best_val_nll) else None,
            "epochs_run": len(rows),
        }

    @torch.no_grad()
    def _predict_proba(self, idx: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.ds is not None, "_predict_proba called before _fit/prepare"
        self.model.eval()
        chunks: list[np.ndarray] = []
        for s in range(0, len(idx), PREDICT_CHUNK):
            batch = self.ds.make_batch_from_indices(idx[s: s + PREDICT_CHUNK], device=self.device)
            out = self.model(batch, sample=False)
            if "P" in out:
                P = out["P"].detach().cpu().to(torch.float64).numpy()
            else:
                V = out["V"].detach().cpu().to(torch.float64).numpy()
                av = batch["av"].detach().cpu().to(torch.float64).numpy()
                P = probs_from_utilities(V, av)
            chunks.append(P)
        return np.concatenate(chunks, axis=0)

    def _config_snapshot(self) -> dict[str, Any]:
        bench = self.model_kwargs.get("cfg")
        return {
            "backend": "torch",
            "model_class": self.model_class.__name__,
            "model_config": dataclasses.asdict(self.mcfg),
            "train_config": dataclasses.asdict(self.tcfg_template),
            **({"benchmark_config": bench} if bench else {}),
        }
