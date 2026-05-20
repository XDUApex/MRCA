# MRCA

MRCA is a two-stage root cause analysis pipeline used in our paper artifact:
- Stage 1: anomaly detection from logs and traces
- Stage 2: metric-based root cause localization

This repository keeps runnable code, measured experiment reports, raw profiling evidence, and a single top-level `result/` directory for final deliverables.

## Environment

```bash
source /home/fuxian/miniconda3/etc/profile.d/conda.sh
conda activate MRCA
```

## Main entry points

For artifact reproduction, prefer the canonical config-driven entrypoint below.
The lower-level commands remain available for debugging individual stages.

```bash
python scripts/reproduce_rq1.py --dry-run
python scripts/reproduce_rq1.py --stage stage2 --skip-existing
python scripts/reproduce_rq1.py --dry-run --skip-existing --manifest result/final_submission/final_reproduction_manifest.json
```

The reproduction plan is stored in [configs/final_rq1_mrca.json](configs/final_rq1_mrca.json).
It records the small set of dataset-compatible MRCA execution profiles used for
the final RQ1 table. This keeps the public artifact stable without exposing the
historical tuning runs as the recommended workflow.

`--skip-existing` is useful for artifact inspection because the repository can
ship measured intermediate outputs. Remove it to rerun the selected stages.

### Stage 1

```bash
python -m MRCA.anomaly_detection --dataset tt --modality all --experiment rq1
```

Supported stage-1 modalities:
- `log`
- `trace`
- `all`

### Stage 2

```bash
python -m MRCA.rcl_unified --dataset tt --modality-combo all+metric --experiment rq1
```

Supported stage-2 modality combinations:
- `log+metric`
- `trace+metric`
- `all+metric`

Useful options:

```bash
python -m MRCA.rcl_unified --dataset tt --modality-combo all+metric --granger-only --legacy --experiment rq1
python -m MRCA.rcl_unified --dataset ob --modality-combo all+metric --window-minutes 15 --max-lag 1 --scoring fisher --experiment rq1
```

## Data preprocessing

For TT / OB:

```bash
python -m MRCA.trace_processing --dataset tt --experiment rq1
python -m MRCA.log_processing --dataset tt --experiment rq1
python -m MRCA.data_aggregation --dataset tt --experiment rq1
python -m MRCA.data_conversion --dataset tt --experiment rq1
```

Full preprocessing expects the original datasets to be available at the paths
configured in [config.py](config.py), or through the dataset-specific
environment variables defined there.

## Evaluation

Unified evaluation:

```bash
python evaluation_unified.py --datasets tt,ob,gaia,aiops --experiment rq1 --mode both --output-dir reports/evaluation_all
```

## Final deliverables

The canonical final outputs now live under [result/](result/):

- [result/final_submission/](result/final_submission/)
  - `final_submission.csv`
  - `final_fault_type_results_long.csv`
  - `final_fault_type_results_pivot.csv`
  - `final_overall_results.csv`
  - `final_results_manifest.json`
- [result/rq2/MRCA/](result/rq2/MRCA/)
  - per-dataset/per-modality RQ2 CSVs
  - merged RQ2 result CSV

Generate them with:

```bash
python scripts/build_artifact_outputs.py
```

Use `python scripts/reproduce_rq1.py --dry-run` to inspect the canonical
commands. A full rerun requires the original datasets; the committed final
tables are regenerated from retained measured summaries and profiling evidence.

## Result semantics

### Accuracy policy

- RQ1 reproduction is driven by `configs/final_rq1_mrca.json`.
- Dataset-specific settings are treated as data-format execution profiles: OB/TT legacy metric-pair Granger, GAIA outflow Granger with timestamp/resampling handling, and AIOps service-level outflow for long-table metrics.
- Historical tuning folders are not part of the cleaned artifact.

### RQ2 field semantics

RQ2 numeric cells are numeric in the result CSVs and follow the 20-column schema in [HOW_TO_FILL_CSV.md](HOW_TO_FILL_CSV.md):

- Cost fields come from profiling JSONs.
- `mrr`, `top1`, `top3`, and `top5` are retained measured values, not copied from paper approximations.

## Important measured-source directories

- [reports/stage1_all_KL/](reports/stage1_all_KL/) — measured stage-1 results for all datasets
- [reports/tt_ob_full_rerun/](reports/tt_ob_full_rerun/) — measured TT / OB rerun results
- [reports/final_stage2_fault_type_summary.csv](reports/final_stage2_fault_type_summary.csv) — retained measured stage-2 fault-type results
- [reports/final_mrr_summary.csv](reports/final_mrr_summary.csv) — retained measured MRR values for RQ2
- [anomaly_detection/rq1_anomaly_score/](anomaly_detection/rq1_anomaly_score/) — retained stage-1 service rankings
- [rq2_profiles/](rq2_profiles/) — profiling JSONs used to fill RQ2 cost tables truthfully

These are supporting evidence directories. They are not the final submission package themselves.

## Directory guide

```text
MRCA/
├── MRCA/                    # core pipeline code: preprocessing, anomaly detection, RCA
├── scripts/                 # final reproduction and artifact generation helpers
├── reports/                 # retained measured summaries used to regenerate final tables
├── rq2_profiles/            # raw profiling evidence used to populate RQ2 cost fields
├── result/                  # canonical final deliverables for submission and inspection
│   ├── final_submission/    # final RQ1-style summary tables and manifest
│   └── rq2/MRCA/            # final RQ2 CSVs
├── evaluation_unified.py    # unified evaluation entrypoint
├── HOW_TO_FILL_CSV.md       # RQ2 formatting rules
└── config.py                # repository-level configuration
```

### Why there are many nested folders

- `reports/` keeps only compact measured summaries needed by the final artifact builder.
- `rq2_profiles/` is separate because it stores raw runtime/resource measurement evidence rather than evaluation scores.
- `result/` is the only directory intended for external delivery. If someone only wants the final artifact outputs, they should start there.
