#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 MIMIC_III_RAW_DIR PROCESSED_DIR [NOTEEVENTS_CSV_OR_GZ]" >&2
  exit 2
fi

raw_dir="$(cd "$1" && pwd)"
mkdir -p "$2"
processed_dir="$(cd "$2" && pwd)"
noteevents="${3:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${script_dir}/.." && pwd)"

if [[ -z "${noteevents}" ]]; then
  if [[ -f "${raw_dir}/NOTEEVENTS.csv.gz" ]]; then
    noteevents="${raw_dir}/NOTEEVENTS.csv.gz"
  else
    noteevents="${raw_dir}/NOTEEVENTS.csv"
  fi
else
  noteevents="$(cd "$(dirname "${noteevents}")" && pwd)/$(basename "${noteevents}")"
fi

mkdir -p "${processed_dir}/root"
cd "${script_dir}/mimic3"
python3 -m mimic3benchmark.scripts.extract_subjects "${raw_dir}" "${processed_dir}/root"
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
python3 data_processing/mimic3/build_notes.py \
  --noteevents "${noteevents}" --ehr-root "${processed_dir}"

echo "MIMIC-III processing complete: ${processed_dir}"
