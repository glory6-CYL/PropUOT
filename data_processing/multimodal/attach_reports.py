#!/usr/bin/env python3
"""Attach sectioned radiology reports to a CXR/EHR mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SECTIONS = ("impression", "findings", "last_paragraph", "comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = pd.read_csv(args.mapping)
    reports = pd.read_csv(args.sections)
    reports["study_id"] = reports["study"].astype(str).str.removeprefix("s").astype(int)
    reports = reports[["study_id", *SECTIONS]].drop_duplicates("study_id", keep="last")
    result = mapping.merge(
        reports,
        on="study_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if result["_merge"].ne("both").any():
        missing = int(result["_merge"].ne("both").sum())
        raise ValueError(
            f"{missing} selected studies have no report; rebuild the CXR mapping "
            "with --report-sections before attaching reports"
        )
    result = result.drop(columns="_merge")
    result[list(SECTIONS)] = result[list(SECTIONS)].fillna("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    with_text = result[list(SECTIONS)].ne("").any(axis=1).sum()
    print(f"Saved {len(result):,} rows ({with_text:,} with report text) to {args.output}")


if __name__ == "__main__":
    main()
