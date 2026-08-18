#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 {mortality|readmission} {paired|partial} EHR_ROOT OUTPUT_DIR [extra train.py args]" >&2
  exit 2
fi

task="$1"
setting="$2"
ehr_root="$(realpath -m "$3")"
output_dir="$(realpath -m "$4")"
shift 4

case "${task}" in
  mortality) normalizer="${ehr_root}/ihm_ts.normalizer" ;;
  readmission) normalizer="${ehr_root}/readmission_ts.normalizer" ;;
  *) echo "Unknown task: ${task}" >&2; exit 2 ;;
esac

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${release_root}"
python3 train.py \
  --dataset mimic3 \
  --modalities ehr-note \
  --task "${task}" \
  --setting "${setting}" \
  --ehr-data-dir "${ehr_root}" \
  --normalizer-state "${normalizer}" \
  --output-dir "${output_dir}" \
  "$@"
