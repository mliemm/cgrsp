# Data not used after the five-slot filter

Nothing in this folder is read by the simulator, the experiments, the figure script or the validator. The only calibration input is `../rendering_performance_data.csv` (6,234 jobs). These files are kept for provenance and so the filter can be reproduced.

| File | Rows | What it is |
|---|---|---|
| `rendering_performance_data_unfiltered.csv` | 6,627 | The full performance CSV before the filter. Input of `scripts/filter_over_five_concurrency.py`. |
| `rendering_performance_excluded_over5.csv` | 393 | The jobs the filter removed: every job that overlapped a moment with more than five jobs in flight. |
| `combined_rendering_data.jsonl` | 1,230 | The August 2025 records costed at the Serverless Flex rates (see ch4 of the thesis). Some of these jobs are among the excluded ones. |
| `combined_rendering_data_original.jsonl` | 1,234 | Earlier version of the same August 2025 consolidation. |
| `rendering_performance_data_original.csv` | 1,234 | Earlier version of the August 2025 performance table. |

Regenerate the calibration CSV and the excluded list from the unfiltered file (run from the repository root):

    python3 scripts/filter_over_five_concurrency.py
