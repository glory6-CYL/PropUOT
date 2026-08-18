#!/usr/bin/env python3
"""Validate the processed files required by a PropUOT experiment."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


TASK_DIR = {"mortality": "in-hospital-mortality", "readmission": "readmission"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic3", "mimic4"), required=True)
    parser.add_argument("--task", choices=tuple(TASK_DIR), required=True)
    parser.add_argument(
        "--modalities", choices=("ehr-note", "ehr-cxr", "ehr-cxr-note"), required=True
    )
    parser.add_argument("--ehr-data-dir", type=Path, required=True)
    parser.add_argument("--cxr-data-dir", type=Path)
    parser.add_argument("--cxr-image-dir", type=Path)
    parser.add_argument("--normalizer-state", type=Path, required=True)
    return parser.parse_args()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    task = args.ehr_data_dir / TASK_DIR[args.task]
    counts = {}
    for split in ("train", "val", "test"):
        listfile = task / f"{split}_listfile.csv"
        _require(listfile)
        frame = pd.read_csv(listfile)
        required = {"stay", "period_length", "stay_id", "y_true"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{listfile} is missing columns: {sorted(missing)}")
        timeseries = task / ("test" if split == "test" else "train")
        _require(timeseries)
        missing_examples = [name for name in frame["stay"].head(100) if not (timeseries / name).is_file()]
        if missing_examples:
            raise FileNotFoundError(f"Missing example time-series: {missing_examples[0]}")
        counts[split] = len(frame)

    _require(args.normalizer_state)
    with args.normalizer_state.open("rb") as handle:
        state = pickle.load(handle, encoding="latin1")
    if len(state.get("means", [])) != 76 or len(state.get("stds", [])) != 76:
        raise ValueError("Normalizer must contain 76 means and standard deviations")

    if args.modalities == "ehr-note":
        for partition in ("train", "test"):
            _require(args.ehr_data_dir / f"{partition}_text_fixed")
            _require(args.ehr_data_dir / f"{partition}_starttime.pkl")
        for split in ("train", "val", "test"):
            _require(task / f"{split}_note_listfile.csv")
    else:
        if args.cxr_data_dir is None:
            raise ValueError("--cxr-data-dir is required for CXR modalities")
        filename = (
            "mimic-cxr-note-ehr-split.csv"
            if args.modalities == "ehr-cxr-note"
            else "mimic-cxr-ehr-split.csv"
        )
        mapping_path = args.cxr_data_dir / filename
        _require(mapping_path)
        mapping = pd.read_csv(mapping_path)
        required = {"dicom_id", "study_id", "subject_id", "split", "stay_id", "stay"}
        if args.modalities == "ehr-cxr-note":
            required |= {"impression", "findings", "last_paragraph", "comparison"}
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(f"{mapping_path} is missing columns: {sorted(missing)}")
        image_root = args.cxr_image_dir or args.cxr_data_dir
        _require(image_root)
        print("CXR-linked:", mapping["split"].value_counts().to_dict())
    print("EHR:", counts)
    print("Processed data validation passed")


if __name__ == "__main__":
    main()
