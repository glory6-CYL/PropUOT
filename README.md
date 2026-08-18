# PropUOT

Official PyTorch implementation of **PropUOT** (Propensity-Guided Unbalanced Optimal Transport) for multimodal clinical prediction with MIMIC-III and MIMIC-IV.

## Setup

```bash
conda env create -f environment.yml
conda activate propuot
```

Run all commands below from the repository root. Training uses a single GPU.

## Data

Apply for credentialed access and download:

- [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/)
- [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/)
- [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MIMIC-CXR reports v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/)

Raw and processed MIMIC data are not included in this repository.

### MIMIC-III

```bash
bash data_processing/prepare_mimic3.sh \
  /path/to/MIMIC-III \
  data/processed/mimic3
```

This prepares the EHR time series, IHM and 30-day readmission tasks, normalizers, and clinical notes.

### MIMIC-IV and MIMIC-CXR

```bash
bash data_processing/prepare_mimic4.sh \
  /path/to/MIMIC-IV \
  data/processed/mimic4

mkdir -p data/processed/mimic-cxr

python data_processing/multimodal/build_cxr_ehr_split.py \
  --ehr-task-dir data/processed/mimic4/in-hospital-mortality \
  --all-stays data/processed/mimic4/root/all_stays.csv \
  --cxr-metadata /path/to/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv.gz \
  --output data/processed/mimic-cxr/mimic-cxr-ehr-split.csv
```

For the tri-modal EHR+CXR+report setting:

```bash
python data_processing/multimodal/extract_report_sections.py \
  --reports-dir /path/to/MIMIC-CXR/files \
  --output data/processed/mimic-cxr/mimic_cxr_sectioned.csv

python data_processing/multimodal/build_cxr_ehr_split.py \
  --ehr-task-dir data/processed/mimic4/in-hospital-mortality \
  --all-stays data/processed/mimic4/root/all_stays.csv \
  --cxr-metadata /path/to/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv.gz \
  --report-sections data/processed/mimic-cxr/mimic_cxr_sectioned.csv \
  --output data/processed/mimic-cxr/mimic-cxr-note-ehr-split.csv
```

## Train

MIMIC-III EHR+note:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/mimic3.sh \
  mortality partial \
  data/processed/mimic3 \
  outputs/mimic3_mortality_partial
```

MIMIC-IV EHR+CXR:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/mimic4.sh \
  mortality partial ehr-cxr \
  data/processed/mimic4 \
  data/processed/mimic-cxr \
  outputs/mimic4_mortality_partial \
  --cxr-image-dir /path/to/MIMIC-CXR-JPG
```

MIMIC-IV EHR+CXR+report (fully matched):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train/mimic4.sh \
  mortality paired ehr-cxr-note \
  data/processed/mimic4 \
  data/processed/mimic-cxr \
  outputs/mimic4_mortality_trimodal \
  --cxr-image-dir /path/to/MIMIC-CXR-JPG
```

Following the CMCM convention, an optional pretrained CXR backbone can be placed at `checkpoints/cxr/best_checkpoint.pth.tar` and loaded by appending:

```bash
--load-state-cxr checkpoints/cxr/best_checkpoint.pth.tar
```

Replace `mortality` with `readmission` for 30-day readmission.

Append `--smoke-test` to run one optimization step and verify that training starts correctly. Full training writes `best_checkpoint.pth.tar` and `last_checkpoint.pth.tar` to the output directory.

## Test

MIMIC-III:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test/mimic3.sh \
  mortality \
  outputs/mimic3_mortality_partial/best_checkpoint.pth.tar \
  data/processed/mimic3 \
  outputs/mimic3_mortality_partial/test
```

MIMIC-IV:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test/mimic4.sh \
  mortality \
  outputs/mimic4_mortality_partial/best_checkpoint.pth.tar \
  data/processed/mimic4 \
  data/processed/mimic-cxr \
  outputs/mimic4_mortality_partial/test \
  --cxr-image-dir /path/to/MIMIC-CXR-JPG
```

MIMIC-IV EHR+CXR+report (fully matched):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test/mimic4.sh \
  mortality \
  outputs/mimic4_mortality_trimodal/best_checkpoint.pth.tar \
  data/processed/mimic4 \
  data/processed/mimic-cxr \
  outputs/mimic4_mortality_trimodal/test \
  --cxr-image-dir /path/to/MIMIC-CXR-JPG
```

Append `--smoke-test` to evaluate one batch and verify checkpoint loading and inference.

## Acknowledgements

This implementation was developed with reference to the publicly available [CMCM](https://github.com/HKU-MedAI/CMCM) and [MedFuse](https://github.com/nyuad-cai/MedFuse) codebases. We thank their authors for making their implementations publicly available.

The data-processing pipeline includes code adapted from the MIT-licensed [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) and [MIT-LCP MIMIC-CXR](https://github.com/MIT-LCP/mimic-cxr) tools. The corresponding copyright and license notices are provided in `data_processing/third_party_licenses/`.
