# Reproducibility Package: CGRSP Thesis

This document is the reproducibility package for the thesis *Cloud GPU Rendering
Scheduling Problem (CGRSP)*. It describes the code, datasets, parameter settings,
and the exact procedure to regenerate every number, table, and figure in the
thesis from raw simulation, plus how to verify the result is correct.

All reported numbers are generated, not hand-typed: every statistic in the thesis
prose is a LaTeX macro defined in `thesis/generated/stats.tex`, and every results
table is a generated fragment in `thesis/generated/`. These fragments are produced
from the simulation JSON in `CGRSP/Outputs/` by `generate_data.py`, and
`validate_numbers.py` cross-checks that the macros, prose, and JSON all agree.

---

## 1. Environment

| Component | Version used for the reported results |
|---|---|
| OS | macOS (Darwin 25.5.0) for the reported results; the package was also verified on Windows under WSL2 and with native Windows Python 3.13.15, and on Linux (Ubuntu 22.04) with Python 3.12.16. The code is pure Python and platform-independent. |
| Python | 3.14.0 (minimum 3.11 for the pinned dependencies) |
| numpy | 2.4.2 |
| scipy | 1.17.0 |
| pandas | 3.0.0 (imported only by the Phase 2 report module; touches no reported number) |
| matplotlib | 3.7+ (figures only) |
| TeX | XeLaTeX (TeX Live / MacTeX); the thesis uses `fontspec` and must be built with `xelatex`, not `pdflatex` |

Windows: run the verification under WSL2 or Git Bash. The `.sh` and `.py` entry points are checked out with LF endings through `.gitattributes`, and `validate_numbers.py` forces UTF-8 reads and output. If you run it from native PowerShell on a legacy code page, set `PYTHONUTF8=1` as well. An existing Windows checkout created before this rule needs a fresh clone or `git checkout-index -a -f` to rewrite its line endings. The reference environment for the reported numbers is still macOS with Python 3.14.0; a Windows or WSL run with the pinned dependencies is a portability check, not the original environment. The verification run used Python 3.12.3 in WSL2 and Python 3.13.15 on native Windows.

Install the Python dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r CGRSP/requirements.txt
```

The reported results were produced with the versions above. Numerical results are
seed-pinned (Section 4), so re-running with the same versions reproduces them
byte for byte. Minor patch-version differences in numpy/scipy are not expected to
change results because all randomness flows through explicitly seeded generators.

---

## 2. Calibration provenance (not runtime inputs)

| Dataset | Path | Description |
|---|---|---|
| Rendering performance calibration | `data/rendering_performance_data.csv` | 6,234 real rendering jobs logged by the author's RunPod Serverless rendering pipeline (6,030 production, 203 development-tagged, 1 untagged in the environment column) across 4 GPU types, after excluding the 393 jobs that ran while more than five jobs were in flight (`scripts/filter_over_five_concurrency.py` regenerates this file from `data/unused_after_filter/rendering_performance_data_unfiltered.csv`; the excluded rows are in `data/unused_after_filter/rendering_performance_excluded_over5.csv`). Used to fit the per-stratum log-normal execution-time model. Its cost columns list the Pods securePrice values the simulator uses, not production bills (see data/README.md). |
| GPU pricing | hardcoded `PRICING_LOOKUP` in `CGRSP/cgrsp/data/execution_times.py` | Hourly rental prices (RunPod Secure Cloud, March 2026): $0.46 / $0.27 / $0.25 / $0.25 for 3090 / A5000 / A4500 / A4000. Derived offline from `data/runpod_pricing/distributions_*.json` (`securePrice`). The raw snapshot archive is gitignored, but `data/runpod_pricing_summary/` tracks the minimum columns the two pricing figures need; rebuild it from the archive with `python3 scripts/export_pricing_summary.py`. The simulator reads the hardcoded table, not the snapshots. The repo-root `runpod_gpu_pricing.json` is a separate, unused Flex-tier artifact. |

The empirical execution-time table (W by complexity and GPU stratum, sparse strata estimated) is built in `CGRSP/cgrsp/data/execution_times.py`; the
availability model in `CGRSP/cgrsp/data/runpod_availability.py`. No live cloud
access is required: the simulator is fully self-contained and calibrated offline
from the CSV above.

---

## 3. Parameter Settings

The headline regime is the **hectic day under capacity saturation**:

| Parameter | Value | Where set |
|---|---|---|
| Concurrency limit `C_max` | 5 (concurrency budget; equals RunPod default per-account worker quota) | experiment scripts; swept over {3,5,7,10} in the C_max sensitivity study |
| Arrival rate `lambda` | 0.100 jobs/s (~950 jobs/day) | `CGRSP/main.py`, phase scripts |
| Deadline split | 20% tight (1 h) / 80% loose (8 h) | job generator |
| GPU fleet | 4 types: RTX 3090, A5000, A4000, A4500 | `CGRSP/cgrsp/config.py` |
| Phase 3 replications | 30 independent seeds (0 to 29) | `phase3_statistical.py` (`N_SEEDS=30`) |
| Classical and learned baselines | 30 seeds | `phase3_baselines.py` (workload imported from `phase3_statistical.py`) |
| Fleet-composition sweep | 10 seeds, six five-slot mixes | `phase3_fleet_mix.py` |
| Day-type / deadline / reservation sweeps | 30 seeds | `phase3_sensitivity.py`, `phase3_deadline.py`, `phase3_reservation_sensitivity.py`, `phase3_beta_sensitivity.py`, `phase3_capacity_bound.py`, `phase3_prior_robustness.py` (`N_SEEDS=30`) |
| Parameter-tuning sweeps (CR\*, threshold, C_max, weight, pressure) | 10 seeds | `phase3_{cr,threshold,cmax,weight,pressure}_sensitivity.py` (`N_SEEDS=10`) |
| Estimated-cell and label/speed brackets | 10 seeds | `phase3_sparse_sensitivity.py`, `phase3_kappa_speed_sensitivity.py` (`N_SEEDS=10`) |
| Phase 4 monthly | 30 days x 10 replications | `phase4_monthly.py` |
| CADR critical-ratio threshold `CR*` | 3 (swept in CR sensitivity) | `phase3_cr_sensitivity.py` |
| SPT+Rescue laxity threshold `theta` | 600 s (swept in threshold sensitivity) | `phase3_threshold_sensitivity.py` |

---

## 4. Determinism

Simulation results were historically not reproducible across runs because the
provisioning-delay model seeded a local RNG from Python's `hash()`, which is
randomised per process. This is fixed two ways:

1. **Source fix.** Provisioning-delay seeding uses `zlib.crc32` (process-
   independent), so results no longer depend on the interpreter hash seed.
2. **Pin.** `CGRSP/_repro.py` re-executes the process with `PYTHONHASHSEED=0`.
   It is imported first by every experiment entry point (`main.py` and all
   `phase*.py`). Do not strip the `import _repro` line.

Consequence: with the pin in place, all schedulers reproduce byte for byte. If a
code change to one scheduler shifts another scheduler's numbers, that is a
determinism regression and must be investigated before committing.

---

## 5. One-Command Reproduction

From the repository root:

```bash
./run_all.sh
```

This runs every experiment, regenerates the thesis data fragments and figures,
validates all numbers, builds the thesis PDF, and copies it to `public/thesis.pdf`.
Override the interpreter if needed:

```bash
PYTHON=python3.12 ./run_all.sh
```

Expect on the order of an hour or more end to end on an 8-core laptop; the
per-step wall times are recorded in each output JSON (`metadata.elapsed_sec` or
`runtime_s`) and drift with the code, so they are not frozen here. The largest
single steps are the arrival-rate prior sweep (about 18 min), the day-type
sensitivity and the deadline sweep (about 9 min each). The classical and learned
baselines add about 4 to 5 minutes and the six-mix fleet-composition sweep about 7
minutes on the reference machine; the latter runs ten seeds and is a sensitivity
check, not a second headline experiment. The availability-model sweep runs its 1,500 independent cells across a
process pool, which takes it from about 6 hours to about 7 minutes; each cell
builds its own seeded job stream, fleet and simulator, so the pooled result is
identical to the sequential one. On success the script prints `all checks
passed`. The snapshot rug of `diurnal_model.pdf` and all of `availability_history.pdf` are rendered from `data/runpod_pricing_summary/`, a compact tracked summary of the gitignored `data/runpod_pricing` archive. A fresh clone can therefore regenerate both figures without the archive; rebuild the summary from the archive with `python3 scripts/export_pricing_summary.py` after a new collection. If both the archive and the summary are absent, `generate_figures.py` skips the two figures and keeps the committed PDFs.
Every simulator input is tracked. To rebuild only the PDF
from the current figures and tables:

```bash
./scripts/build_thesis.sh
```

(The script runs BibTeX and three subsequent XeLaTeX passes to settle citations,
cross-references, the contents, and the lists of figures, tables, and algorithms.
It copies the finished PDF to `public/thesis.pdf` only after a successful build.)

---

## 6. Pipeline Map (experiment -> output -> thesis artifact)

Each experiment script writes a JSON file; `generate_data.py` reads those JSON
files and writes the `thesis/generated/` fragments consumed by the `.tex`
chapters.

| Experiment script | Output JSON | Feeds |
|---|---|---|
| `CGRSP/main.py` | `Outputs/runpod_scheduler/report.json` | Phase 2 table, figures; its `verification` block backs the Ch4 constraint-verifier claim for the seven baselines |
| `CGRSP/verify_proposed.py` (run after `main.py`) | `Outputs/runpod_scheduler/verification_proposed.json` | Ch4 constraint-verifier claim for RH, CADR and Adaptive (checked by `validate_numbers.py`) |
| `CGRSP/main.py --jobs 100 --arrival-rate 0.002 --output CGRSP/Outputs/runpod_normal --seed 42` | `Outputs/runpod_normal/report.json` | Phase 2 figure, normal-day panel |
| `CGRSP/phase3_statistical.py` | `Outputs/phase3/statistical.json` | Phase 3 stats table, t-tests, ablation, headline macros |
| `CGRSP/phase3_baselines.py` | `Outputs/phase3/baselines.json` | Classical and learned baseline table (Section 5.4.2) |
| `CGRSP/phase3_fleet_mix.py` | `Outputs/phase3/fleet_mix.json` | Fleet-composition sensitivity table (Section 5.5.12) |
| `CGRSP/phase3_sensitivity.py` | `Outputs/phase3/sensitivity.json` | Day-type sensitivity table, heatmap |
| `CGRSP/phase3_deadline.py` | `Outputs/phase3/deadline.json` | Deadline-tightness table, crossover prose |
| `CGRSP/phase3_cmax_sensitivity.py` | `Outputs/phase3/cmax_sensitivity.json` | C_max sensitivity table |
| `CGRSP/phase3_threshold_sensitivity.py` | `Outputs/phase3/threshold_sensitivity.json` | Rescue-threshold table |
| `CGRSP/phase3_beta_sensitivity.py` | `Outputs/phase3/beta_sensitivity.json` | Availability-model (beta_g) sensitivity table |
| `CGRSP/phase3_capacity_bound.py` | `Outputs/phase3/capacity_bound.json` | Capacity lower bound + clairvoyant relaxation macros |
| `CGRSP/phase3_prior_robustness.py` | `Outputs/phase3/prior_robustness.json` | Reversed-rank beta_g arm, RH prior-ablation and arrival-rate prior macros |
| `CGRSP/phase3_pressure_sensitivity.py` | `Outputs/phase3/pressure_sensitivity.json` | Adaptive queue-pressure (P) sweep macros |
| `CGRSP/phase3_cr_sensitivity.py` | `Outputs/phase3/cr_sensitivity.json` | CADR CR* sensitivity table |
| `CGRSP/phase3_reservation_sensitivity.py` | `Outputs/phase3/reservation_sensitivity.json` | RH anticipation payoff prose |
| `CGRSP/phase3_sparse_sensitivity.py` | `Outputs/phase3/sparse_sensitivity.json` | Estimated-cell exposure and service-model sensitivity |
| `CGRSP/phase3_kappa_speed_sensitivity.py` | `Outputs/phase3/kappa_speed_sensitivity.json` | Complexity-label noise and A4000 speed brackets |
| `CGRSP/phase3_diagnostics.py` | `Outputs/phase3/diagnostics.json` | Mean-based service rerun (30 seeds), RH t_free collapse share, per-seed tight misses and dispatch clock |
| `CGRSP/phase3_weight_sensitivity.py` | `Outputs/phase3/weight_sensitivity.json` | Objective-weight sensitivity table (w_cost, alpha2/alpha1, tardiness variant) |
| `CGRSP/phase4_monthly.py` | `Outputs/phase4/monthly_results.json` | Phase 4 monthly table, figure, macros |

`phase3_fleet_mix.py` runs ten seeds over six five-slot mixes. Only the
all-types anchor reproduces the first ten seeds of the Phase 3 record exactly
(360 of 360 compared values across the six focal policies and six metrics); the
other five mixes are sensitivity checks at `n=10`, and a different fleet can
change which slot is free at each dispatch, so their service draws need not
match the anchor seed by seed. The sweep is not a second studio trace.

Generators and validation:

| Step | Command | Produces |
|---|---|---|
| Thesis data fragments | `python3 generate_data.py` | `thesis/generated/*.tex` (tables + `stats.tex`) |
| Figures | `python3 thesis/figures/generate_figures.py` and `python3 generate_gantt.py` | `thesis/figures/*.pdf` (also copied into `paper/`, created on first run) |
| Validation | `python3 validate_numbers.py` | exits 0 only if macros, prose, and JSON all agree |

---

## 7. Verifying a Reproduction

After `./run_all.sh`, confirm:

1. `validate_numbers.py` exits 0 with `all checks passed`.
2. `git diff CGRSP/Outputs` shows changes only to the timing fields
   `metadata.elapsed_sec`, `metadata.generated_at` and the top-level `runtime_s`
   in `beta_sensitivity.json`, `pressure_sensitivity.json` and
   `prior_robustness.json`. Those record wall-clock timing and necessarily differ
   between runs, so a byte-for-byte criterion cannot be met. Any diff in a
   *results* field is a determinism regression.
3. The thesis builds with `xelatex` and no undefined references.

The PDF includes a separate title page after the cover. For the paper copy,
insert the signed advisor recommendation and committee approval sheets between
the cover and title page after printing. Before library upload, insert their
legible A4 scans in the same position so the first three PDF pages are the
cover, recommendation, and approval, as required by the
[NTUST library checklist](https://libraryfile.lib.ntust.edu.tw/training/guide/ChartChecklist.pdf).
The generated PDF has no manually applied watermark.

To regenerate only the derived artifacts from existing JSON (no experiments):

```bash
python3 generate_data.py && python3 validate_numbers.py
```

---

## 8. Code Layout

| Path | Role |
|---|---|
| `CGRSP/cgrsp/simulator/` | Discrete-event simulator core, fleet, metrics |
| `CGRSP/cgrsp/schedulers/` | The 10 scheduling policies (FIFO, SPT, EDF, LCF, Balanced, Random, SPT+Rescue, Adaptive, RH, CADR) |
| `CGRSP/cgrsp/data/` | Execution-time model, job generator, availability model |
| `CGRSP/phase*.py`, `CGRSP/main.py` | Experiment entry points (each pins the seed via `_repro`) |
| `generate_data.py` | Builds `thesis/generated/` from `CGRSP/Outputs/` |
| `validate_numbers.py` | Cross-checks every reported number, convention, and spelling |
| `thesis/` | LaTeX source; build with `xelatex` |

The monthly experiment uses one persistent fleet and queue across the calendar. Each day's candidate Poisson trace is cut at 86,400 seconds; excluded arrivals are recorded, accepted jobs carry across day boundaries, and the final backlog is drained. Monthly means use actual accepted/completed job counts, and monthly p95 is a nearest-rank percentile of pooled per-job tardiness. The objective-weight experiment uses mean wait in seconds and missed-job fractions before normalisation.

