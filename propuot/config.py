from __future__ import annotations

import argparse
from pathlib import Path


TASK_DIRS = {
    "mortality": "in-hospital-mortality",
    "readmission": "readmission",
}

DEFAULT_SEED = 13
DEFAULT_EHR_LAYERS = 2
DEFAULT_EHR_DROPOUT = 0.2


def _common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=("mimic3", "mimic4"))
    parser.add_argument(
        "--modalities",
        required=True,
        choices=("ehr-note", "ehr-cxr", "ehr-cxr-note"),
    )
    parser.add_argument("--task", required=True, choices=tuple(TASK_DIRS))
    parser.add_argument("--setting", required=True, choices=("paired", "partial"))
    parser.add_argument("--ehr-data-dir", required=True)
    parser.add_argument("--cxr-data-dir")
    parser.add_argument(
        "--cxr-image-dir",
        help="MIMIC-CXR-JPG root (defaults to --cxr-data-dir)",
    )
    parser.add_argument("--normalizer-state", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PropUOT on a processed MIMIC cohort")
    _common_data_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", help="Resume from a PropUOT checkpoint")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real optimization step and exit without training an epoch",
    )

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--ehr-layers", type=int, default=DEFAULT_EHR_LAYERS)
    parser.add_argument("--ehr-dropout", type=float, default=DEFAULT_EHR_DROPOUT)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--vision-backbone", default="resnet34", choices=("resnet34",))
    parser.add_argument(
        "--pretrained-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--load-state-cxr",
        "--load_state_cxr",
        dest="load_state_cxr",
        help="Optional CMCM-style checkpoint used to initialize the CXR backbone",
    )
    parser.add_argument("--text-model", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--text-hidden-size", type=int, default=312)
    parser.add_argument("--text-output-size", type=int, default=512)
    parser.add_argument("--max-text-length", type=int, default=512)

    parser.add_argument(
        "--lambda-prop",
        type=float,
        default=None,
        help="Propensity-loss weight (default: 0.05 for partial, 0 for paired)",
    )
    parser.add_argument("--lambda-uot", type=float, default=0.05)
    # GeomLoss backend parameters.
    parser.add_argument("--sinkhorn-blur", type=float, default=0.10)
    parser.add_argument("--sinkhorn-reach", type=float, default=0.10)
    parser.add_argument("--omega-gamma", type=float, default=0.50)
    parser.add_argument("--propensity-eps", type=float, default=1e-6)
    parser.add_argument("--propensity-bound", type=float, default=0.05)
    parser.add_argument("--omega-max", type=float, default=10.0)
    parser.add_argument("--propensity-entropy-weight", type=float, default=0.005)
    parser.add_argument(
        "--propensity-entropy-sign",
        type=float,
        default=-1.0,
        choices=(-1.0, 1.0),
    )
    parser.add_argument("--temperature-rate", type=float, default=1e-4)
    parser.add_argument("--temperature-min", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    return parser


def build_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained PropUOT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ehr-data-dir", required=True)
    parser.add_argument("--cxr-data-dir")
    parser.add_argument("--cxr-image-dir")
    parser.add_argument("--normalizer-state", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate one real batch to verify checkpoint loading and inference",
    )
    return parser


def finalize_train_args(args: argparse.Namespace) -> argparse.Namespace:
    _validate_experiment(args)
    args.task_dir = TASK_DIRS[args.task]
    if args.lambda_prop is None:
        args.lambda_prop = 0.05 if args.setting == "partial" else 0.0
    if args.batch_size is None:
        args.batch_size = 16 if args.dataset == "mimic3" else 32
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for UOT alignment")
    if args.load_state_cxr and "cxr" not in args.modalities:
        raise ValueError("--load-state-cxr is only valid for CXR experiments")
    if not (0.0 < args.propensity_bound < 0.5):
        raise ValueError("--propensity-bound must lie in (0, 0.5)")
    return args


def merge_test_with_checkpoint(
    cli_args: argparse.Namespace, checkpoint_config: dict
) -> argparse.Namespace:
    if not checkpoint_config:
        raise ValueError("Checkpoint does not contain a saved PropUOT configuration")
    merged = dict(checkpoint_config)
    for key in (
        "ehr_data_dir",
        "cxr_data_dir",
        "cxr_image_dir",
        "normalizer_state",
        "num_workers",
        "device",
        "seed",
    ):
        value = getattr(cli_args, key, None)
        if value is not None:
            merged[key] = value
    if cli_args.batch_size is not None:
        merged["batch_size"] = cli_args.batch_size
    merged["checkpoint"] = cli_args.checkpoint
    merged["bootstrap_samples"] = cli_args.bootstrap_samples
    merged["smoke_test"] = bool(cli_args.smoke_test)
    merged["output_dir"] = cli_args.output_dir or str(Path(cli_args.checkpoint).resolve().parent)
    args = argparse.Namespace(**merged)
    _validate_experiment(args)
    args.task_dir = TASK_DIRS[args.task]
    return args


def _validate_experiment(args: argparse.Namespace) -> None:
    if args.dataset == "mimic3" and args.modalities != "ehr-note":
        raise ValueError("MIMIC-III experiments support --modalities ehr-note")
    if args.dataset == "mimic4" and args.modalities not in {"ehr-cxr", "ehr-cxr-note"}:
        raise ValueError("MIMIC-IV experiments support ehr-cxr or ehr-cxr-note")
    if "cxr" in args.modalities and not getattr(args, "cxr_data_dir", None):
        raise ValueError("--cxr-data-dir is required for CXR experiments")
    if args.modalities == "ehr-cxr-note" and args.setting != "paired":
        raise ValueError("Tri-modal experiments require --setting paired")
