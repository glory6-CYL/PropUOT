import argparse
import hashlib
import os
import shutil


def move_to_partition(root, patients, partition):
    destination_root = os.path.join(root, partition)
    os.makedirs(destination_root, exist_ok=True)
    for patient in patients:
        shutil.move(
            os.path.join(root, patient),
            os.path.join(destination_root, patient),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create a deterministic patient-level MIMIC-IV train/test split."
    )
    parser.add_argument("subjects_root_path")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must lie in (0, 1)")

    patients = sorted(filter(str.isdigit, os.listdir(args.subjects_root_path)))
    ranked = sorted(
        patients,
        key=lambda patient: hashlib.sha256(
            f"{args.seed}:{patient}".encode("utf-8")
        ).digest(),
    )
    test_count = round(args.test_fraction * len(ranked))
    test_set = set(ranked[:test_count])
    train = [patient for patient in patients if patient not in test_set]
    test = [patient for patient in patients if patient in test_set]
    move_to_partition(args.subjects_root_path, train, "train")
    move_to_partition(args.subjects_root_path, test, "test")
    print(f"Patient split: train={len(train):,}, test={len(test):,}")


if __name__ == "__main__":
    main()
