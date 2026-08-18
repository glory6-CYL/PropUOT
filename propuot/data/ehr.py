from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


DEFAULT_CONFIG = Path(__file__).parent / "resources" / "discretizer_config.json"


class Discretizer:
    def __init__(
        self,
        timestep: float = 2.0,
        store_masks: bool = True,
        impute_strategy: str = "previous",
        start_time: str = "zero",
        config_path: str | Path = DEFAULT_CONFIG,
    ) -> None:
        with Path(config_path).open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.id_to_channel = config["id_to_channel"]
        self.channel_to_id = {name: index for index, name in enumerate(self.id_to_channel)}
        self.is_categorical = config["is_categorical_channel"]
        self.possible_values = config["possible_values"]
        self.normal_values = config["normal_values"]
        self.header = ["Hours", *self.id_to_channel]
        self.timestep = float(timestep)
        self.store_masks = bool(store_masks)
        self.start_time = start_time
        self.impute_strategy = impute_strategy
        if self.impute_strategy not in {"zero", "normal_value", "previous", "next"}:
            raise ValueError(f"Unsupported imputation strategy: {self.impute_strategy}")

    @property
    def output_header(self) -> list[str]:
        names: list[str] = []
        for channel in self.id_to_channel:
            if self.is_categorical[channel]:
                names.extend(f"{channel}->{value}" for value in self.possible_values[channel])
            else:
                names.append(channel)
        if self.store_masks:
            names.extend(f"mask->{channel}" for channel in self.id_to_channel)
        return names

    def transform(
        self, values: np.ndarray, header: list[str] | None = None, end: float | None = None
    ) -> tuple[np.ndarray, str]:
        header = header or self.header
        if not len(values):
            raise ValueError("Encountered an empty EHR time-series")
        if header[0] != "Hours":
            raise ValueError("The first EHR time-series column must be Hours")

        eps = 1e-6
        times = [float(row[0]) for row in values]
        if any(a >= b + eps for a, b in zip(times, times[1:])):
            raise ValueError("EHR events must be sorted by time")
        first_time = times[0] if self.start_time == "relative" else 0.0
        if self.start_time not in {"relative", "zero"}:
            raise ValueError(f"Unsupported start time: {self.start_time}")
        max_hours = (max(times) if end is None else float(end)) - first_time
        bin_count = int(max_hours / self.timestep + 1.0 - eps)
        if bin_count <= 0:
            raise ValueError(f"Invalid observation window: {max_hours} hours")

        starts: list[int] = []
        cursor = 0
        for channel in self.id_to_channel:
            starts.append(cursor)
            cursor += len(self.possible_values[channel]) if self.is_categorical[channel] else 1
        data = np.zeros((bin_count, cursor), dtype=np.float64)
        observed = np.zeros((bin_count, len(self.id_to_channel)), dtype=np.int8)
        originals = [["" for _ in self.id_to_channel] for _ in range(bin_count)]

        def write(bin_index: int, channel: str, value: str) -> None:
            channel_index = self.channel_to_id[channel]
            output_index = starts[channel_index]
            if self.is_categorical[channel]:
                try:
                    category = self.possible_values[channel].index(value)
                except ValueError as exc:
                    raise ValueError(f"Unknown categorical value {value!r} for {channel}") from exc
                category_count = len(self.possible_values[channel])
                data[bin_index, output_index : output_index + category_count] = 0.0
                data[bin_index, output_index + category] = 1.0
            else:
                data[bin_index, output_index] = float(value)

        for row in values:
            time = float(row[0]) - first_time
            if time > max_hours + eps:
                continue
            bin_index = int(time / self.timestep - eps)
            if not 0 <= bin_index < bin_count:
                continue
            for column, raw in enumerate(row[1:], start=1):
                if raw == "":
                    continue
                channel = header[column]
                channel_index = self.channel_to_id[channel]
                observed[bin_index, channel_index] = 1
                originals[bin_index][channel_index] = raw
                write(bin_index, channel, raw)

        if self.impute_strategy in {"normal_value", "previous"}:
            previous: list[str | None] = [None] * len(self.id_to_channel)
            for bin_index in range(bin_count):
                for channel in self.id_to_channel:
                    channel_index = self.channel_to_id[channel]
                    if observed[bin_index, channel_index]:
                        previous[channel_index] = originals[bin_index][channel_index]
                        continue
                    value = self.normal_values[channel]
                    if self.impute_strategy == "previous" and previous[channel_index] is not None:
                        value = previous[channel_index]
                    write(bin_index, channel, value)
        elif self.impute_strategy == "next":
            following: list[str | None] = [None] * len(self.id_to_channel)
            for bin_index in range(bin_count - 1, -1, -1):
                for channel in self.id_to_channel:
                    channel_index = self.channel_to_id[channel]
                    if observed[bin_index, channel_index]:
                        following[channel_index] = originals[bin_index][channel_index]
                        continue
                    write(
                        bin_index,
                        channel,
                        following[channel_index] or self.normal_values[channel],
                    )

        if self.store_masks:
            data = np.hstack((data, observed.astype(np.float32)))
        return data, ",".join(self.output_header)


class Normalizer:
    def __init__(self, fields: list[int]) -> None:
        self.fields = fields
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        self.means = np.asarray(payload["means"])
        self.stds = np.asarray(payload["stds"])

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer parameters have not been loaded")
        result = values.astype(np.float64, copy=True)
        standard_deviations = self.stds[self.fields].copy()
        standard_deviations[standard_deviations < 1e-7] = 1.0
        result[:, self.fields] = (
            result[:, self.fields] - self.means[self.fields]
        ) / standard_deviations
        return result


class EHRDataset(Dataset):
    def __init__(
        self,
        discretizer: Discretizer,
        normalizer: Normalizer,
        listfile: str | Path,
        timeseries_dir: str | Path,
        period_length: float = 48.0,
    ) -> None:
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.timeseries_dir = Path(timeseries_dir)
        self.period_length = float(period_length)
        with Path(listfile).open(encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines:
            raise ValueError(f"Empty EHR listfile: {listfile}")
        header = lines[0].strip().split(",")
        self.label_index = header.index("y_true") if "y_true" in header else 3
        self.stay_id_index = header.index("stay_id") if "stay_id" in header else 2
        self.records: dict[str, dict[str, float]] = {}
        for raw in lines[1:]:
            columns = raw.rstrip("\n").split(",")
            if not columns or not columns[0]:
                continue
            self.records[columns[0]] = {
                "period": float(columns[1]),
                "stay_id": float(columns[self.stay_id_index]),
                "label": float(columns[self.label_index]),
            }
        self.names = list(self.records)
        self.CLASSES = ["y_true"]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int | str) -> tuple[np.ndarray, float]:
        name = self.names[index] if isinstance(index, int) else index
        record = self.records[name]
        path = self.timeseries_dir / name
        rows: list[np.ndarray] = []
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            for raw in handle:
                columns = raw.rstrip("\n").split(",")
                if columns and float(columns[0]) <= self.period_length + 1e-6:
                    rows.append(np.asarray(columns))
        if not rows:
            raise ValueError(f"No events in the first 48 hours: {path}")
        period = record["period"] if record["period"] > 0 else self.period_length
        values = self.discretizer.transform(np.stack(rows), header=header, end=period)[0]
        return self.normalizer.transform(values), record["label"]


def build_ehr_datasets(args) -> tuple[EHRDataset, EHRDataset, EHRDataset]:
    root = Path(args.ehr_data_dir)
    task_dir = root / args.task_dir
    required = (
        task_dir / "train_listfile.csv",
        task_dir / "val_listfile.csv",
        task_dir / "test_listfile.csv",
        Path(args.normalizer_state),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing processed EHR files:\n  " + "\n  ".join(missing))

    discretizer = Discretizer(timestep=2.0, store_masks=True, impute_strategy="previous")
    continuous = [
        index for index, name in enumerate(discretizer.output_header) if "->" not in name
    ]
    normalizer = Normalizer(continuous)
    normalizer.load(args.normalizer_state)
    datasets = (
        EHRDataset(discretizer, normalizer, task_dir / "train_listfile.csv", task_dir / "train"),
        EHRDataset(discretizer, normalizer, task_dir / "val_listfile.csv", task_dir / "train"),
        EHRDataset(discretizer, normalizer, task_dir / "test_listfile.csv", task_dir / "test"),
    )
    feature_count = len(discretizer.output_header)
    if feature_count != 76:
        raise ValueError(f"Expected 76 EHR features after discretization, found {feature_count}")
    return datasets
