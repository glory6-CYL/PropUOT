#!/usr/bin/env python3
"""Fit PropUOT's 2-hour EHR normalizer on a task's training split."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from propuot.data.ehr import Discretizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Processed in-hospital-mortality or readmission directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-samples", type=int, default=-1, help="Use -1 for all training stays"
    )
    return parser.parse_args()


def _records(listfile: Path) -> list[tuple[str, float]]:
    with listfile.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            (row["stay"], float(row.get("period_length", 48.0) or 48.0))
            for row in reader
        ]


def _read_timeseries(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        rows = [np.asarray(line.rstrip("\n").split(",")) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty EHR time-series: {path}")
    return np.stack(rows), header


def main() -> None:
    args = parse_args()
    listfile = args.task_dir / "train_listfile.csv"
    timeseries_dir = args.task_dir / "train"
    records = _records(listfile)
    if args.max_samples >= 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No training stays were selected")

    discretizer = Discretizer(
        timestep=2.0, store_masks=True, impute_strategy="previous", start_time="zero"
    )
    feature_count = len(discretizer.output_header)
    total = np.zeros(feature_count, dtype=np.float64)
    total_squared = np.zeros(feature_count, dtype=np.float64)
    row_count = 0
    for stay, period in tqdm(records, desc="Fitting EHR normalizer"):
        raw, header = _read_timeseries(timeseries_dir / stay)
        end = period if period > 0 else 48.0
        values, _ = discretizer.transform(raw, header=header, end=end)
        total += values.sum(axis=0, dtype=np.float64)
        total_squared += np.square(values, dtype=np.float64).sum(axis=0)
        row_count += values.shape[0]

    if row_count < 2:
        raise ValueError("At least two time bins are required to fit a normalizer")
    means = total / row_count
    centered_sum_squared = total_squared - 2.0 * total * means + row_count * np.square(means)
    variances = np.maximum(centered_sum_squared / (row_count - 1), 0.0)
    standard_deviations = np.sqrt(variances)
    standard_deviations[standard_deviations < 1e-7] = 1e-7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(
            {"means": means, "stds": standard_deviations},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        f"Saved a {feature_count}-feature normalizer fitted on "
        f"{len(records):,} stays ({row_count:,} time bins) to {args.output}"
    )


if __name__ == "__main__":
    main()
