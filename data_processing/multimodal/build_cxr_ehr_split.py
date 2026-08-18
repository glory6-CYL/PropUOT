#!/usr/bin/env python3
"""Deterministically link MIMIC-CXR images to processed MIMIC-IV stays."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPORT_SECTIONS = ("impression", "findings", "last_paragraph", "comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ehr-task-dir",
        type=Path,
        required=True,
        help="Processed task directory containing train/val/test_listfile.csv",
    )
    parser.add_argument(
        "--all-stays",
        type=Path,
        required=True,
        help="all_stays.csv produced by the MIMIC-IV extractor",
    )
    parser.add_argument(
        "--cxr-metadata",
        type=Path,
        required=True,
        help="mimic-cxr-2.0.0-metadata.csv.gz (or unpacked CSV)",
    )
    parser.add_argument(
        "--report-sections",
        type=Path,
        help="Optional sectioned-report CSV used to build a fully matched cohort",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--views",
        nargs="+",
        default=["AP"],
        choices=("AP", "PA"),
        help="Frontal view positions to include",
    )
    parser.add_argument(
        "--selection",
        choices=("closest", "latest"),
        default="latest",
        help="Choose the closest-to-admission or latest eligible image",
    )
    parser.add_argument("--window-hours", type=float, default=48.0)
    return parser.parse_args()


def _ehr_stays(task_dir: Path) -> pd.DataFrame:
    frames = []
    for filename, split in (
        ("train_listfile.csv", "train"),
        ("val_listfile.csv", "validate"),
        ("test_listfile.csv", "test"),
    ):
        path = task_dir / filename
        frame = pd.read_csv(path, usecols=["stay_id", "stay"])
        frame["split"] = split
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["stay_id"] = result["stay_id"].astype("int64")
    return result.drop_duplicates("stay_id")


def _study_datetime(frame: pd.DataFrame) -> pd.Series:
    def normalize_time(value: object) -> str:
        if pd.isna(value):
            return "000000"
        return f"{int(float(value)):06d}"

    time = frame["StudyTime"].map(normalize_time)
    return pd.to_datetime(
        frame["StudyDate"].astype(str) + " " + time,
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )


def _load_report_sections(path: Path) -> pd.DataFrame:
    reports = pd.read_csv(path)
    missing = sorted(set(REPORT_SECTIONS) - set(reports.columns))
    if missing:
        raise ValueError(f"{path} is missing report columns: {', '.join(missing)}")
    if "study_id" not in reports.columns:
        if "study" not in reports.columns:
            raise ValueError(f"{path} must contain study or study_id")
        reports["study_id"] = pd.to_numeric(
            reports["study"].astype(str).str.removeprefix("s"), errors="raise"
        )
    reports["study_id"] = reports["study_id"].astype("int64")
    return reports[["study_id", *REPORT_SECTIONS]].drop_duplicates(
        "study_id", keep="last"
    )


def build_mapping(args: argparse.Namespace) -> pd.DataFrame:
    ehr = _ehr_stays(args.ehr_task_dir)
    stays = pd.read_csv(
        args.all_stays, usecols=["subject_id", "stay_id", "intime"]
    )
    stays["stay_id"] = stays["stay_id"].astype("int64")
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays = stays.merge(ehr, on="stay_id", how="inner", validate="one_to_one")

    metadata = pd.read_csv(
        args.cxr_metadata,
        usecols=[
            "dicom_id",
            "study_id",
            "subject_id",
            "ViewPosition",
            "StudyDate",
            "StudyTime",
        ],
        low_memory=False,
    )
    metadata["study_datetime"] = _study_datetime(metadata)
    report_sections = getattr(args, "report_sections", None)
    reports = _load_report_sections(report_sections) if report_sections else None
    if reports is not None:
        metadata = metadata.loc[metadata["study_id"].isin(reports["study_id"])]
    merged = metadata.merge(stays, on="subject_id", how="inner")
    merged = merged.dropna(subset=["study_datetime", "intime"])
    merged = merged.loc[merged["ViewPosition"].isin(args.views)].copy()
    merged["hours_from_intime"] = (
        (merged["study_datetime"] - merged["intime"]).dt.total_seconds() / 3600.0
    )
    merged = merged.loc[
        merged["hours_from_intime"].between(0.0, args.window_hours, inclusive="both")
    ]
    if args.selection == "closest":
        matched = merged.loc[
            merged.groupby("stay_id")["hours_from_intime"].idxmin()
        ]
    else:
        matched = (
            merged.sort_values("study_datetime", kind="stable")
            .groupby("stay_id", sort=False)
            .tail(1)
        )
    matched = matched.loc[
        :,
        [
            "dicom_id",
            "study_id",
            "subject_id",
            "split",
            "stay_id",
            "stay",
        ],
    ]
    matched = matched.sort_values(["split", "subject_id", "stay_id"])
    if matched["stay_id"].duplicated().any():
        raise RuntimeError("CXR matching produced duplicate ICU stays")
    if reports is not None:
        expected = len(matched)
        matched = matched.merge(
            reports, on="study_id", how="inner", validate="many_to_one"
        )
        if len(matched) != expected:
            raise RuntimeError("Report matching changed the linked ICU-stay cohort")
        matched[list(REPORT_SECTIONS)] = matched[list(REPORT_SECTIONS)].fillna("")
    return matched.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    matched = build_mapping(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output, index=False)
    print(f"Saved {len(matched):,} linked stays to {args.output}")
    print(matched["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
