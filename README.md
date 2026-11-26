# CGRSP: Cloud GPU Rendering Scheduling Problem

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](CGRSP/requirements.txt)
[![Thesis PDF](https://img.shields.io/badge/thesis-PDF-red.svg)](thesis/thesis.pdf)

Code, data and results for the master's thesis **Cloud GPU Rendering Scheduling Problem (CGRSP): A Mathematical Formulation and Empirical Evaluation on Heterogeneous GPU Fleets** (Department of Computer Science and Information Engineering, National Taiwan University of Science and Technology).

Author: Michael (林墾森). Advisors: Prof. Yuan-Shin Hwang (黃元欣) and Prof. Vincent F. Yu (喻奉天).

## Abstract

CGRSP assigns batch 3D rendering jobs, one frame per job, to heterogeneous on-demand cloud GPUs under uncertain node availability and mixed deadlines. Waiting time and missed soft deadlines are minimised first, rental cost second. A discrete-event simulator, calibrated on 6,234 recorded RunPod Serverless rendering jobs across four GPU types, compares ten scheduling policies over four experimental phases. Two schedulers are proposed: an anticipative Rolling-Horizon scheduler (RH) and a Cost-Aware Deadline-Risk scheduler (CADR). Under a hectic load of about 950 jobs per day and a five-slot concurrency limit, RH and CADR miss the fewest deadlines, with no significant paired difference between them, and CADR's mean tardiness is far lower than RH's. GPU speeds come from measured execution records; per-type availability is an assumed stock-status model, so all scheduling results are simulation-based.

## Key results

Phase 3, hectic day (about 950 jobs, $C_{\max}=5$, 20% tight 1 h and 80% loose 8 h deadlines), mean over 30 seeds. Full table with 95% confidence intervals: Table 5.2 of the thesis.

| Scheduler | Mean wait (min) | Deadline miss (%) | Mean tardiness (min) |
|---|---:|---:|---:|
| **RH** (proposed) | 141.70 | **9.60** | 31.84 |
| **CADR** (proposed) | 144.66 | 9.68 | 5.95 |
| SPT+Rescue | 146.32 | 13.56 | 5.89 |
| Adaptive | 169.67 | 12.62 | 4.91 |
| SPT | **141.45** | 18.96 | 12.02 |
| EDF | 170.22 | 14.71 | 5.16 |
| FIFO | 167.93 | 24.43 | 26.01 |
| LCF | 166.91 | 24.24 | 25.25 |
| Balanced | 168.59 | 25.25 | 26.66 |
| Random | 167.68 | 23.44 | 32.89 |

The RH and CADR miss rates do not differ significantly (paired $t$-test, $p = 0.912$). The thesis recommends CADR as the default and RH for scarce-availability settings where the count of missed deadlines matters most.

## Installation

Requires Python 3.11 or newer. The reported results were produced with Python 3.14.0 and the pinned versions in `CGRSP/requirements.txt`. Building the thesis PDF needs XeLaTeX (TeX Live or MacTeX).

```bash
git clone https://github.com/mliemm/cgrsp.git
cd cgrsp
python3 -m pip install -r CGRSP/requirements.txt
```

## Reproducing the results

```bash
# Check every number reported in the thesis against the generated outputs
python3 validate_numbers.py

# Run the test suite (207 tests)
python3 -m pytest

# Rebuild the thesis PDF from the committed tables and figures (about 1 minute)
./scripts/build_thesis.sh

# Rerun every experiment, regenerate all tables and figures, and rebuild the PDF (about 1 hour)
./run_all.sh

# Verify a fresh clone: install, tests, number validation and result hashes
./scripts/verify_fresh_clone.sh --fast
```

All runs are deterministic. Every experiment entry point imports `CGRSP/_repro.py`, which pins `PYTHONHASHSEED=0`, and all randomness flows through explicitly seeded generators, so a rerun reproduces the result files byte for byte. `REPRODUCE.md` maps each experiment script to its output file and to the table or figure it feeds, and lists the parameter settings.

## Repository structure

```text
.
├── CGRSP/                  Simulator, schedulers, experiment drivers and tests
│   ├── cgrsp/              Simulator package (fleet, availability model, schedulers, metrics)
│   ├── main.py             Phase 2 benchmark (single seed)
│   ├── phase3_*.py         Phase 3 statistical validation and sensitivity sweeps
│   ├── phase4_monthly.py   Phase 4 monthly simulation
│   ├── tests/              Test suite
│   └── Outputs/            Result files (JSON) behind every reported number
├── data/                   Calibration dataset and pricing summary
├── thesis/                 LaTeX source, figures, generated tables and thesis.pdf
├── scripts/                Build, data-filtering and verification scripts
├── docs/                   Verification report, result hash manifest, revision-marked thesis
├── generate_data.py        Builds the thesis tables and statistics macros from Outputs/
├── validate_numbers.py     Cross-checks the thesis text against the result files
├── run_all.sh              One-command full reproduction
└── REPRODUCE.md            Detailed reproducibility guide
```

`docs/thesis_revisions_highlighted.pdf` is the final thesis with every change since the 9 November 2026 defense marked (additions in blue, deletions in red).

## Data

`data/rendering_performance_data.csv` holds 6,234 rendering jobs logged by the author's RunPod Serverless pipeline between July 2025 and June 2026, across four GPU types (RTX 3090, RTX A5000, RTX A4500, RTX A4000). Each row has the job timestamps, GPU type, render time, numeric material and model identifiers, and an environment tag; it contains no client names or billing details. The 393 jobs that ran while more than five jobs were in flight are excluded and kept in `data/unused_after_filter/`. The raw RunPod pricing snapshots are not redistributed because of their size; `data/runpod_pricing_summary/` keeps the columns needed to regenerate the two pricing figures. See `data/README.md` for the column definitions.

## Citation

If you use this code or data, please cite the thesis:

```bibtex
@mastersthesis{michael2027cgrsp,
  author  = {Michael},
  title   = {Cloud {GPU} Rendering Scheduling Problem ({CGRSP}): A Mathematical Formulation and Empirical Evaluation on Heterogeneous {GPU} Fleets},
  school  = {National Taiwan University of Science and Technology},
  address = {Taipei, Taiwan},
  year    = {2027}
}
```

GitHub's "Cite this repository" button reads the same metadata from `CITATION.cff`.

## License

The code in `CGRSP/`, `scripts/` and `thesis/figures/` is released under the MIT License (see `LICENSE`). The thesis text, the built PDF and the dataset in `data/` are copyright 2026 Michael and are provided for academic reproducibility of this thesis.

## Acknowledgements

The discrete-event core of the simulator is adapted from SPARS (Amrizal et al., *SoftwareX* 34:102693, 2026, https://doi.org/10.1016/j.softx.2026.102693), which is MIT-licensed; see `CGRSP/THIRD_PARTY_NOTICES.md`.
