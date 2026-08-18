#!/usr/bin/env python3
"""Optionally create a flat, width-resized MIMIC-CXR cache."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Optional CXR/EHR mapping; when set, cache only its dicom_id values",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def _resize(job: tuple[Path, Path, int]) -> tuple[Path, str | None]:
    source, destination, width = job
    if destination.is_file():
        return destination, None
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            height = max(1, round(image.height * width / image.width))
            image.resize((width, height), Image.Resampling.BILINEAR).save(
                destination, quality=95
            )
        return destination, None
    except Exception as exc:  # report corrupt files without aborting the cache job
        return source, str(exc)


def main() -> None:
    args = parse_args()
    if args.width < 224:
        raise ValueError("--width must be at least 224")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.images_dir.glob("p??/p*/s*/*.jpg"))
    if not sources:
        raise FileNotFoundError(f"No MIMIC-CXR JPG files found under {args.images_dir}")
    if args.mapping is not None:
        with args.mapping.open(newline="", encoding="utf-8") as handle:
            selected = {row["dicom_id"] for row in csv.DictReader(handle)}
        sources = [path for path in sources if path.stem in selected]
        if len(sources) != len(selected):
            raise FileNotFoundError(
                f"Found {len(sources):,}/{len(selected):,} mapped images under {args.images_dir}"
            )
    jobs = [(path, args.output_dir / path.name, args.width) for path in sources]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for path, error in tqdm(executor.map(_resize, jobs), total=len(jobs)):
            if error:
                failures.append((path, error))
    print(f"Processed {len(jobs) - len(failures):,}/{len(jobs):,} images")
    if failures:
        print(f"Warning: {len(failures):,} images failed; first failure: {failures[0]}")


if __name__ == "__main__":
    main()
