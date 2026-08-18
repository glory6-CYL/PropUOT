#!/usr/bin/env python3
"""Build the MIMIC-III note modality used by PropUOT.

The script never copies source data into the repository. It joins NOTEEVENTS
to the benchmark ICU episodes, retains notes recorded before the 48-hour
prediction time, and writes one derived JSON file per episode.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Episode:
    name: str
    partition: str
    start: pd.Timestamp

    @property
    def prediction_time(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(hours=48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PropUOT's MIMIC-III note files and sampled listfiles"
    )
    parser.add_argument(
        "--noteevents",
        type=Path,
        required=True,
        help="MIMIC-III NOTEEVENTS.csv or NOTEEVENTS.csv.gz",
    )
    parser.add_argument(
        "--ehr-root",
        type=Path,
        required=True,
        help="Processed MIMIC-III root containing root/ and task directories",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("in-hospital-mortality", "readmission"),
        help="Task to process; repeat for both (default: both)",
    )
    parser.add_argument("--sample-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing derived note files created by an earlier run",
    )
    return parser.parse_args()


def _episode_parts(stay: str) -> tuple[str, int]:
    match = re.fullmatch(r"(\d+)_episode(\d+)_timeseries\.csv", stay)
    if not match:
        raise ValueError(f"Unexpected benchmark stay name: {stay}")
    return match.group(1), int(match.group(2))


def _load_episodes(ehr_root: Path, tasks: list[str]) -> tuple[dict[int, Episode], dict]:
    root = ehr_root / "root"
    by_admission: dict[int, Episode] = {}
    starts = {"train": {}, "test": {}}
    seen_stays: set[tuple[str, str]] = set()

    for task in tasks:
        task_dir = ehr_root / task
        for split in ("train", "val", "test"):
            listfile = task_dir / f"{split}_listfile.csv"
            if not listfile.is_file():
                raise FileNotFoundError(f"Missing benchmark listfile: {listfile}")
            frame = pd.read_csv(listfile, usecols=["stay"])
            partition = "test" if split == "test" else "train"
            for stay in frame["stay"].astype(str):
                if (partition, stay) in seen_stays:
                    continue
                seen_stays.add((partition, stay))
                subject, episode_number = _episode_parts(stay)
                stays_path = root / partition / subject / "stays.csv"
                if not stays_path.is_file():
                    raise FileNotFoundError(f"Missing per-subject stays file: {stays_path}")
                stays = pd.read_csv(stays_path).sort_values("INTIME").reset_index(drop=True)
                index = episode_number - 1
                if not 0 <= index < len(stays):
                    raise IndexError(f"{stay} does not have a matching row in {stays_path}")
                row = stays.iloc[index]
                note_name = f"{subject}_{episode_number}"
                episode = Episode(note_name, partition, pd.Timestamp(row["INTIME"]))
                hadm_id = int(row["HADM_ID"])
                previous = by_admission.get(hadm_id)
                if previous is not None and previous != episode:
                    raise ValueError(f"Admission {hadm_id} maps to multiple ICU episodes")
                by_admission[hadm_id] = episode
                starts[partition][note_name] = episode.start
    return by_admission, starts


def _normalize_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def _extract_to_database(
    noteevents: Path,
    episodes: dict[int, Episode],
    database: sqlite3.Connection,
    chunksize: int,
) -> int:
    required = [
        "HADM_ID",
        "CHARTDATE",
        "CHARTTIME",
        "STORETIME",
        "ISERROR",
        "TEXT",
    ]
    database.execute(
        "CREATE TABLE notes (episode TEXT, partition TEXT, charttime TEXT, text TEXT)"
    )
    database.execute("CREATE INDEX notes_episode ON notes(partition, episode, charttime)")
    admission_ids = set(episodes)
    kept = 0
    reader = pd.read_csv(
        noteevents,
        usecols=required,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="Reading NOTEEVENTS chunks"):
        chunk = chunk.loc[chunk["HADM_ID"].notna()].copy()
        chunk["HADM_ID"] = chunk["HADM_ID"].astype(int)
        chunk = chunk.loc[chunk["HADM_ID"].isin(admission_ids)]
        if "ISERROR" in chunk:
            error_flag = pd.to_numeric(chunk["ISERROR"], errors="coerce").fillna(0)
            chunk = chunk.loc[error_flag.ne(1)]
        timestamps = chunk["CHARTTIME"].fillna(chunk["STORETIME"]).fillna(chunk["CHARTDATE"])
        chunk = chunk.assign(timestamp=pd.to_datetime(timestamps, errors="coerce"))
        rows: list[tuple[str, str, str, str]] = []
        for row in chunk.itertuples(index=False):
            episode = episodes[int(row.HADM_ID)]
            timestamp = row.timestamp
            if pd.isna(timestamp) or timestamp > episode.prediction_time:
                continue
            text = _normalize_text(row.TEXT)
            if text:
                rows.append(
                    (episode.name, episode.partition, timestamp.isoformat(sep=" "), text)
                )
        if rows:
            database.executemany("INSERT INTO notes VALUES (?, ?, ?, ?)", rows)
            database.commit()
            kept += len(rows)
    return kept


def _write_episode_files(
    ehr_root: Path,
    starts: dict,
    database: sqlite3.Connection,
    overwrite: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for partition in ("train", "test"):
        destination = ehr_root / f"{partition}_text_fixed"
        destination.mkdir(parents=True, exist_ok=True)
        existing = next(destination.iterdir(), None)
        if existing is not None and not overwrite:
            raise FileExistsError(
                f"{destination} is not empty; pass --overwrite to replace matching files"
            )
        written = 0
        for episode in tqdm(sorted(starts[partition]), desc=f"Writing {partition} notes"):
            rows = database.execute(
                "SELECT charttime, text FROM notes "
                "WHERE partition = ? AND episode = ? ORDER BY charttime",
                (partition, episode),
            ).fetchall()
            grouped: dict[str, list[str]] = {}
            for timestamp, text in rows:
                grouped.setdefault(timestamp, []).append(text)
            if not grouped:
                continue
            payload = {timestamp: " ".join(texts) for timestamp, texts in grouped.items()}
            target = destination / episode
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            written += 1
        with (ehr_root / f"{partition}_starttime.pkl").open("wb") as handle:
            pickle.dump(starts[partition], handle, protocol=pickle.HIGHEST_PROTOCOL)
        counts[partition] = written
    return counts


def _write_sampled_listfiles(
    ehr_root: Path, tasks: list[str], fraction: float, seed: int
) -> None:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--sample-fraction must lie in (0, 1]")
    for task in tasks:
        task_dir = ehr_root / task
        for split in ("train", "val", "test"):
            source = task_dir / f"{split}_listfile.csv"
            frame = pd.read_csv(source).sort_values("stay")
            sampled = frame.sample(frac=fraction, random_state=seed)
            target = task_dir / f"{split}_note_listfile.csv"
            sampled.to_csv(target, index=False)
            print(f"Saved {len(sampled):,} sampled stays: {target}")


def main() -> None:
    args = parse_args()
    tasks = args.task or ["in-hospital-mortality", "readmission"]
    episodes, starts = _load_episodes(args.ehr_root, tasks)
    print(f"Loaded {len(episodes):,} unique admissions")
    with tempfile.NamedTemporaryFile(prefix="propuot_notes_", suffix=".sqlite") as handle:
        database = sqlite3.connect(handle.name)
        try:
            kept = _extract_to_database(
                args.noteevents, episodes, database, args.chunksize
            )
            counts = _write_episode_files(
                args.ehr_root, starts, database, args.overwrite
            )
        finally:
            database.close()
    _write_sampled_listfiles(args.ehr_root, tasks, args.sample_fraction, args.seed)
    print(f"Retained {kept:,} note rows; episode files: {counts}")


if __name__ == "__main__":
    main()
