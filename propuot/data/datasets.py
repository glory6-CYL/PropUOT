from __future__ import annotations

import json
import math
import pickle
import random
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from transformers import AutoTokenizer

from .ehr import EHRDataset, build_ehr_datasets


REPORT_SECTIONS = ("impression", "findings", "last_paragraph", "comparison")


def _hours_between(start: str, end: str) -> int:
    delta = np.datetime64(end) - np.datetime64(start)
    return int(delta.astype("timedelta64[h]").astype(np.int64))


class MIMIC3NoteIndex:
    def __init__(self, args, split: str) -> None:
        root = Path(args.ehr_data_dir)
        listfile = root / args.task_dir / f"{split}_note_listfile.csv"
        if not listfile.is_file():
            raise FileNotFoundError(f"Missing note listfile: {listfile}")
        names = pd.read_csv(listfile)["stay"].astype(str).tolist()
        partition = "test" if split == "test" else "train"
        note_dir = root / f"{partition}_text_fixed"
        starttime_path = root / f"{partition}_starttime.pkl"
        if not note_dir.is_dir() or not starttime_path.is_file():
            raise FileNotFoundError(
                f"Missing extracted notes or start times: {note_dir}, {starttime_path}"
            )
        with starttime_path.open("rb") as handle:
            start_times = pickle.load(handle, encoding="latin1")

        self.texts: dict[str, list[str]] = {}
        available = {path.name for path in note_dir.iterdir() if path.is_file()}
        for stay in names:
            tokens = stay.split("_")
            note_name = f"{tokens[0]}_{tokens[1].replace('episode', '')}"
            if note_name not in available or start_times.get(note_name, -1) == -1:
                continue
            with (note_dir / note_name).open(encoding="utf-8") as handle:
                events = json.load(handle)
            selected: list[str] = []
            for timestamp in sorted(events):
                hours = _hours_between(str(start_times[note_name]), timestamp)
                if hours <= 48:
                    value = events[timestamp]
                    selected.append(
                        " ".join(map(str, value))
                        if isinstance(value, list)
                        else str(value)
                    )
                else:
                    break
            # Preserve the cohort construction used by the experiments: a stay
            # must have more than five notes, after which the final five are used.
            if len(selected) <= 5:
                continue
            self.texts[stay] = selected[-5:]
        self.names = list(self.texts)


class CXRIndex:
    def __init__(self, args, split: str, include_reports: bool) -> None:
        self.root = Path(args.cxr_data_dir)
        self.image_root = Path(getattr(args, "cxr_image_dir", None) or args.cxr_data_dir)
        filename = (
            "mimic-cxr-note-ehr-split.csv"
            if include_reports
            else "mimic-cxr-ehr-split.csv"
        )
        split_file = self.root / filename
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing CXR/EHR mapping: {split_file}")
        frame = pd.read_csv(split_file)
        required = {"dicom_id", "study_id", "subject_id", "split", "stay_id", "stay"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{split_file} is missing columns: {', '.join(missing)}")
        split_name = {"train": "train", "val": "validate", "test": "test"}[split]
        frame = frame.loc[frame["split"] == split_name].copy()
        frame["stay"] = frame["stay"].astype(str)
        frame = frame.drop_duplicates(subset="stay_id", keep="first")
        if include_reports:
            if missing := sorted(set(REPORT_SECTIONS) - set(frame.columns)):
                raise ValueError(f"{split_file} is missing report columns: {', '.join(missing)}")
            frame[list(REPORT_SECTIONS)] = frame[list(REPORT_SECTIONS)].fillna("")
        self.rows = {str(row.stay): row for row in frame.itertuples(index=False)}
        self.include_reports = include_reports

    def _image_path(self, row) -> Path:
        dicom_id = str(row.dicom_id)
        subject_id = str(int(row.subject_id))
        study_id = str(int(row.study_id))
        group = f"p{subject_id[:2]}"
        candidates = (
            self.image_root / "resized" / f"{dicom_id}.jpg",
            self.image_root / "files" / group / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg",
            self.image_root / group / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Could not locate {dicom_id}.jpg. Checked:\n  "
            + "\n  ".join(str(path) for path in candidates)
        )

    def load_image(self, stay: str, transform) -> torch.Tensor:
        row = self.rows[stay]
        with Image.open(self._image_path(row)) as image:
            return transform(image.convert("RGB"))

    def reports(self, stay: str) -> list[str]:
        row = self.rows[stay]
        return [str(getattr(row, section)) for section in REPORT_SECTIONS]


class FusionDataset(Dataset):
    CLASSES = ["y_true"]

    def __init__(
        self,
        args,
        ehr: EHRDataset,
        split: str,
        tokenizer=None,
        note_index: MIMIC3NoteIndex | None = None,
        cxr_index: CXRIndex | None = None,
    ) -> None:
        self.args = args
        self.ehr = ehr
        self.split = split
        self.tokenizer = tokenizer
        self.note_index = note_index
        self.cxr_index = cxr_index
        if note_index is not None:
            paired = [name for name in note_index.names if name in ehr.records]
        elif cxr_index is not None:
            paired = [name for name in cxr_index.rows if name in ehr.records]
        else:  # pragma: no cover - guarded by experiment validation
            raise ValueError("A paired modality index is required")
        self.paired_names = paired
        paired_set = set(paired)
        self.paired_set = paired_set
        self.unpaired_names = sorted(name for name in ehr.names if name not in paired_set)
        self.names = (
            self.paired_names
            if args.setting == "paired"
            else [*self.paired_names, *self.unpaired_names]
        )
        self.observed_indices = list(range(len(self.paired_names)))
        if not self.paired_names:
            raise ValueError(f"No paired samples found in the {split} split")
        print(
            f"[{split}] total={len(self.names)} paired={len(self.paired_names)} "
            f"unpaired={len(self.names) - len(self.paired_names)}"
        )

        self.image_transform = _image_transforms(training=split == "train")

    def __len__(self) -> int:
        return len(self.names)

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_text_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def __getitem__(self, index: int) -> dict:
        stay = self.names[index]
        ehr_values, label = self.ehr[stay]
        observed = stay in self.paired_set
        item = {
            "ehr": ehr_values,
            "label": float(label),
            "image": None,
            "input_ids": None,
            "attention_mask": None,
            "presence": None,
        }
        if self.args.modalities == "ehr-note":
            if observed:
                item["input_ids"], item["attention_mask"] = self._tokenize(
                    self.note_index.texts[stay]
                )
            item["presence"] = [observed]
        elif self.args.modalities == "ehr-cxr":
            if observed:
                item["image"] = self.cxr_index.load_image(stay, self.image_transform)
            item["presence"] = [observed]
        else:
            item["image"] = self.cxr_index.load_image(stay, self.image_transform)
            item["input_ids"], item["attention_mask"] = self._tokenize(
                self.cxr_index.reports(stay)
            )
            item["presence"] = [True, True]
        return item


def _image_transforms(training: bool):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    operations: list = [transforms.Resize(256)]
    if training:
        operations.extend(
            (
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=45,
                    scale=(0.85, 1.15),
                    shear=0,
                    translate=(0.15, 0.15),
                ),
            )
        )
    operations.extend((transforms.CenterCrop(224), transforms.ToTensor(), normalize))
    return transforms.Compose(operations)


def _pad_ehr(values: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
    lengths = [array.shape[0] for array in values]
    maximum = max(lengths)
    padded = [
        np.concatenate(
            (array, np.zeros((maximum - array.shape[0], array.shape[1]), dtype=array.dtype)),
            axis=0,
        )
        for array in values
    ]
    return torch.from_numpy(np.stack(padded)).float(), lengths


def collate_fusion(batch: list[dict], modalities: str, max_text_length: int) -> dict:
    ehr, lengths = _pad_ehr([item["ehr"] for item in batch])
    output = {
        "ehr": ehr,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1),
        "presence": torch.tensor([item["presence"] for item in batch], dtype=torch.float32),
        "image": None,
        "input_ids": None,
        "attention_mask": None,
    }
    if "cxr" in modalities:
        output["image"] = torch.stack(
            [
                torch.zeros((3, 224, 224), dtype=torch.float32)
                if item["image"] is None
                else item["image"]
                for item in batch
            ]
        )
    if "note" in modalities:
        note_count = 4 if modalities == "ehr-cxr-note" else 5
        shape = (note_count, max_text_length)
        output["input_ids"] = torch.stack(
            [
                torch.zeros(shape, dtype=torch.long)
                if item["input_ids"] is None
                else item["input_ids"]
                for item in batch
            ]
        )
        output["attention_mask"] = torch.stack(
            [
                torch.zeros(shape, dtype=torch.long)
                if item["attention_mask"] is None
                else item["attention_mask"]
                for item in batch
            ]
        )
    return output


class ObservedBatchSampler(Sampler[list[int]]):
    """Shuffle batches while guaranteeing at least one observed pair per batch."""

    def __init__(
        self, dataset: FusionDataset, batch_size: int, seed: int, drop_last: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        batch_count = len(self)
        if batch_count == 0:
            raise ValueError("Training set is smaller than one batch")
        observed = list(self.dataset.observed_indices)
        if not observed:
            raise ValueError("PropUOT training requires at least one observed modality pair")
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        generator.shuffle(observed)

        anchors = [observed[index % len(observed)] for index in range(batch_count)]
        anchored_unique = set(anchors)
        remaining = [index for index in range(len(self.dataset)) if index not in anchored_unique]
        generator.shuffle(remaining)
        batches = [[anchor] for anchor in anchors]
        cursor = 0
        for batch in batches:
            capacity = self.batch_size - 1
            batch.extend(remaining[cursor : cursor + capacity])
            cursor += capacity
        generator.shuffle(batches)
        for batch in batches:
            if len(batch) == self.batch_size or not self.drop_last:
                generator.shuffle(batch)
                yield batch


def build_dataloaders(args) -> tuple[DataLoader, DataLoader, DataLoader]:
    ehr_datasets = build_ehr_datasets(args)
    tokenizer = None
    if "note" in args.modalities:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)

    fusion_datasets: list[FusionDataset] = []
    for split, ehr in zip(("train", "val", "test"), ehr_datasets):
        if args.dataset == "mimic3":
            note_index = MIMIC3NoteIndex(args, split)
            dataset = FusionDataset(
                args, ehr, split, tokenizer=tokenizer, note_index=note_index
            )
        else:
            cxr_index = CXRIndex(
                args, split, include_reports=args.modalities == "ehr-cxr-note"
            )
            dataset = FusionDataset(
                args, ehr, split, tokenizer=tokenizer, cxr_index=cxr_index
            )
        fusion_datasets.append(dataset)

    collate = partial(
        collate_fusion,
        modalities=args.modalities,
        max_text_length=args.max_text_length,
    )
    train_sampler = ObservedBatchSampler(
        fusion_datasets[0], args.batch_size, args.seed, drop_last=True
    )
    common = {
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        fusion_datasets[0], batch_sampler=train_sampler, **common
    )
    val_loader = DataLoader(
        fusion_datasets[1], batch_size=args.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        fusion_datasets[2], batch_size=args.batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, test_loader
