#!/usr/bin/env python3
"""Extract Impression, Findings, Last paragraph, and Comparison from reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from section_parser import TARGETS, extract_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="MIMIC-CXR report files/ directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = sorted(args.reports_dir.glob("p??/p*/s*.txt"))
    if not reports:
        raise FileNotFoundError(f"No MIMIC-CXR reports found under {args.reports_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study", *TARGETS])
        writer.writeheader()
        for path in tqdm(reports, desc="Parsing reports"):
            sections = extract_sections(
                path.read_text(encoding="utf-8", errors="replace"), path.stem
            )
            writer.writerow({"study": path.stem, **sections})
    print(f"Saved {len(reports):,} sectioned reports to {args.output}")


if __name__ == "__main__":
    main()
