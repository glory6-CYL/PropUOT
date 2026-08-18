from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .metrics import binary_metrics, bootstrap_metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class PropUOTTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, args) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(args.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.start_epoch = 1
        self.best_auroc = float("-inf")
        self.bad_epochs = 0
        self.history: list[dict] = []
        if args.resume:
            self._resume(args.resume)

    def _total_loss(self, output: dict, labels: torch.Tensor, epoch: int) -> tuple:
        temperature = max(
            math.exp(-self.args.temperature_rate * epoch),
            self.args.temperature_min,
        )
        task = F.binary_cross_entropy_with_logits(output["logits"], labels) / temperature
        propensity = self.args.lambda_prop * output["propensity_loss"]
        uot = self.args.lambda_uot * output["uot_loss"]
        return task + propensity + uot, task, propensity, uot

    def smoke_test(self) -> dict[str, float]:
        """Exercise one real training batch, including backward and optimizer step."""
        self.model.train()
        batch = move_batch(next(iter(self.train_loader)), self.device)
        output = self.model(batch)
        loss, task, propensity, uot = self._total_loss(
            output, batch["labels"], epoch=1
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite PropUOT smoke-test loss: {loss}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        metrics = {
            "loss": float(loss.detach()),
            "task_loss": float(task.detach()),
            "propensity_loss": float(propensity.detach()),
            "uot_loss": float(uot.detach()),
            "batch_size": int(batch["labels"].shape[0]),
        }
        self._save_checkpoint("smoke_checkpoint.pth.tar", epoch=0)
        print(
            "PropUOT smoke test passed: "
            + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
            + f" checkpoint={self.output_dir / 'smoke_checkpoint.pth.tar'}",
            flush=True,
        )
        return metrics

    def _run_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "task_loss": 0.0, "propensity_loss": 0.0, "uot_loss": 0.0}
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        for step, batch in enumerate(self.train_loader, start=1):
            batch = move_batch(batch, self.device)
            output = self.model(batch)
            loss, task, propensity, uot = self._total_loss(output, batch["labels"], epoch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task.detach())
            totals["propensity_loss"] += float(propensity.detach())
            totals["uot_loss"] += float(uot.detach())
            labels_all.append(batch["labels"].detach().cpu().numpy())
            probabilities_all.append(torch.sigmoid(output["logits"]).detach().cpu().numpy())
            if step % 100 == 0:
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(self.train_loader):04d} "
                    f"loss={totals['loss'] / step:.5f}",
                    flush=True,
                )

        count = len(self.train_loader)
        result = {key: value / count for key, value in totals.items()}
        result.update(binary_metrics(np.concatenate(labels_all), np.concatenate(probabilities_all)))
        return result

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        losses: list[float] = []
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for batch in self.val_loader:
            batch = move_batch(batch, self.device)
            # Propensity/UOT objectives are training regularizers and needlessly
            # make validation inference much slower.
            output = self.model(batch, compute_auxiliary_losses=False)
            loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"])
            losses.append(float(loss))
            labels.append(batch["labels"].cpu().numpy())
            probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
        result = {"loss": float(np.mean(losses))}
        result.update(binary_metrics(np.concatenate(labels), np.concatenate(probabilities)))
        return result

    def train(self) -> Path:
        config_path = self.output_dir / "config.json"
        config_path.write_text(
            json.dumps(vars(self.args), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            train_metrics = self._run_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step(val_metrics["loss"])
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            self.history.append(record)
            print(
                f"epoch={epoch:03d} train_auc={train_metrics['auroc']:.4f} "
                f"val_auc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f}",
                flush=True,
            )

            improved = val_metrics["auroc"] > self.best_auroc
            if improved:
                self.best_auroc = val_metrics["auroc"]
                self.bad_epochs = 0
                self._save_checkpoint("best_checkpoint.pth.tar", epoch)
            else:
                self.bad_epochs += 1
            self._save_checkpoint("last_checkpoint.pth.tar", epoch)
            (self.output_dir / "history.json").write_text(
                json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
            )
            if self.bad_epochs >= self.args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break
        best = self.output_dir / "best_checkpoint.pth.tar"
        if not best.is_file():
            raise RuntimeError("Training completed without producing a best checkpoint")
        return best

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "format": "propuot-v1",
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_auroc": self.best_auroc,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": vars(self.args),
        }

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        target = self.output_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary)
        temporary.replace(target)

    def _resume(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "propuot-v1":
            raise ValueError(f"Not a PropUOT checkpoint: {checkpoint_path}")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_auroc = float(checkpoint.get("best_auroc", float("-inf")))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        self.history = list(checkpoint.get("history", []))
        print(f"Resuming PropUOT from epoch {self.start_epoch}", flush=True)


@torch.no_grad()
def evaluate_checkpoint(model: nn.Module, test_loader, args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "propuot-v1":
        raise ValueError(f"Not a PropUOT checkpoint: {args.checkpoint}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    propensities: list[np.ndarray] = []
    presence: list[np.ndarray] = []
    for batch_index, batch in enumerate(test_loader):
        batch = move_batch(batch, device)
        output = model(batch, compute_auxiliary_losses=False)
        labels.append(batch["labels"].detach().cpu().numpy())
        probabilities.append(
            torch.sigmoid(output["logits"]).detach().cpu().numpy()
        )
        propensities.append(output["propensity"].detach().cpu().numpy())
        presence.append(batch["presence"].detach().cpu().numpy())
        if args.smoke_test and batch_index == 0:
            break

    labels_array = np.concatenate(labels).reshape(-1)
    probability_array = np.concatenate(probabilities).reshape(-1)
    propensity_array = np.concatenate(propensities).reshape(-1)
    presence_array = np.concatenate(presence)
    metrics = (
        binary_metrics(labels_array, probability_array)
        if args.smoke_test
        else bootstrap_metrics(
            labels_array,
            probability_array,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    metrics.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "samples": int(labels_array.size),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "smoke_test": bool(args.smoke_test),
        }
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        labels=labels_array,
        probabilities=probability_array,
    )
    np.savez_compressed(
        output_dir / "test_propensity.npz",
        propensity=propensity_array,
        presence=presence_array,
    )
    return metrics
