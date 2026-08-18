#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 {mortality|readmission} {paired|partial} {ehr-cxr|ehr-cxr-note} EHR_ROOT CXR_ROOT OUTPUT_DIR [extra train.py args]" >&2
  exit 2
fi

task="$1"
setting="$2"
modalities="$3"
ehr_root="$(realpath -m "$4")"
cxr_root="$(realpath -m "$5")"
output_dir="$(realpath -m "$6")"
shift 6

case "${task}" in
  mortality) normalizer="${ehr_root}/ihm_ts.normalizer" ;;
  readmission) normalizer="${ehr_root}/readmission_ts.normalizer" ;;
  *) echo "Unknown task: ${task}" >&2; exit 2 ;;
esac

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${release_root}"
python3 train.py \
  --dataset mimic4 \
  --modalities "${modalities}" \
  --task "${task}" \
  --setting "${setting}" \
  --ehr-data-dir "${ehr_root}" \
  --cxr-data-dir "${cxr_root}" \
  --normalizer-state "${normalizer}" \
  --output-dir "${output_dir}" \
  "$@"
