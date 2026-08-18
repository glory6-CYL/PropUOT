import argparse
import hashlib
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Create a deterministic patient-level MIMIC-IV train/validation split."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Fraction of post-test training subjects used for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must lie in (0, 1)")

    with open(os.path.join(args.dataset_dir, "train/listfile.csv")) as handle:
        lines = handle.readlines()
    header, examples = lines[0], lines[1:]
    patients = sorted({line.split("_", 1)[0] for line in examples})
    ranked = sorted(
        patients,
        key=lambda patient: hashlib.sha256(
            f"{args.seed}:{patient}".encode("utf-8")
        ).digest(),
    )
    validation_count = round(args.validation_fraction * len(ranked))
    validation_patients = set(ranked[:validation_count])
    train = [line for line in examples if line.split("_", 1)[0] not in validation_patients]
    validation = [line for line in examples if line.split("_", 1)[0] in validation_patients]

    for filename, selected in (
        ("train_listfile.csv", train),
        ("val_listfile.csv", validation),
    ):
        with open(os.path.join(args.dataset_dir, filename), "w") as handle:
            handle.write(header)
            handle.writelines(selected)
    shutil.copy(
        os.path.join(args.dataset_dir, "test/listfile.csv"),
        os.path.join(args.dataset_dir, "test_listfile.csv"),
    )
    print(f"Stay split: train={len(train):,}, validation={len(validation):,}")


if __name__ == "__main__":
    main()
