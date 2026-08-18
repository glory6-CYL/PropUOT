from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .config import (
    build_test_parser,
    build_train_parser,
    finalize_train_args,
    merge_test_with_checkpoint,
)
from .data import build_dataloaders
from .models.propuot import build_propuot
from .training import PropUOTTrainer, evaluate_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_main() -> None:
    args = finalize_train_args(build_train_parser().parse_args())
    seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _ = build_dataloaders(args)
    model = build_propuot(args)
    trainer = PropUOTTrainer(model, train_loader, val_loader, args)
    if args.smoke_test:
        trainer.smoke_test()
        return
    checkpoint = trainer.train()
    print(f"Best checkpoint: {checkpoint}")


def test_main() -> None:
    cli_args = build_test_parser().parse_args()
    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    args = merge_test_with_checkpoint(cli_args, checkpoint.get("config", {}))
    # The checkpoint supplies the learned vision weights; avoid downloading a
    # second ImageNet copy merely to overwrite it immediately.
    args.pretrained_vision = False
    args.load_state_cxr = None
    args.resume = None
    seed_everything(args.seed)
    _, _, test_loader = build_dataloaders(args)
    model = build_propuot(args)
    metrics = evaluate_checkpoint(model, test_loader, args)
    prefix = "PropUOT test smoke passed: " if args.smoke_test else ""
    print(
        f"{prefix}AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"(n={metrics['samples']})"
    )
