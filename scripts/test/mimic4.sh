#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 {mortality|readmission} CHECKPOINT EHR_ROOT CXR_ROOT OUTPUT_DIR [extra test.py args]" >&2
  exit 2
fi

task="$1"
checkpoint="$(realpath -m "$2")"
ehr_root="$(realpath -m "$3")"
cxr_root="$(realpath -m "$4")"
output_dir="$(realpath -m "$5")"
shift 5

case "${task}" in
  mortality) normalizer="${ehr_root}/ihm_ts.normalizer" ;;
  readmission) normalizer="${ehr_root}/readmission_ts.normalizer" ;;
  *) echo "Unknown task: ${task}" >&2; exit 2 ;;
esac

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${release_root}"
python3 test.py \
  --checkpoint "${checkpoint}" \
  --ehr-data-dir "${ehr_root}" \
  --cxr-data-dir "${cxr_root}" \
  --normalizer-state "${normalizer}" \
  --output-dir "${output_dir}" \
  "$@"
