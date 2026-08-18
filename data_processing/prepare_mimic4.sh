#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MIMIC_IV_RAW_DIR PROCESSED_DIR" >&2
  exit 2
fi

raw_dir="$(cd "$1" && pwd)"
mkdir -p "$2"
processed_dir="$(cd "$2" && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${script_dir}/.." && pwd)"

mkdir -p "${processed_dir}/root"
cd "${script_dir}/mimic4"
python3 -m mimic3benchmark.scripts.extract_subjects_iv "${raw_dir}" "${processed_dir}/root"
python3 -m mimic3benchmark.scripts.validate_events "${processed_dir}/root"
python3 -m mimic3benchmark.scripts.extract_episodes_from_subjects "${processed_dir}/root"
python3 -m mimic3benchmark.scripts.split_train_and_test "${processed_dir}/root"
python3 -m mimic3benchmark.scripts.create_in_hospital_mortality \
  "${processed_dir}/root" "${processed_dir}/in-hospital-mortality"
python3 -m mimic3benchmark.scripts.create_readmission_30d \
  --root_path "${processed_dir}/root" --output_path "${processed_dir}/readmission"
python3 -m mimic3models.split_train_val "${processed_dir}/in-hospital-mortality"
python3 -m mimic3models.split_train_val "${processed_dir}/readmission"

cd "${release_root}"
python3 data_processing/fit_normalizer.py \
  --task-dir "${processed_dir}/in-hospital-mortality" \
  --output "${processed_dir}/ihm_ts.normalizer"
python3 data_processing/fit_normalizer.py \
  --task-dir "${processed_dir}/readmission" \
  --output "${processed_dir}/readmission_ts.normalizer"

echo "MIMIC-IV EHR processing complete: ${processed_dir}"
