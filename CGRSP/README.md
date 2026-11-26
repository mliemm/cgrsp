# CGRSP Simulator

Cloud GPU Rendering Scheduling Problem (CGRSP): a discrete-event simulation framework for evaluating scheduling algorithms on an on-demand heterogeneous GPU cloud (RunPod Secure Cloud). GPU instances are provisioned per job, billed per second, and terminated on completion. The operator sets a concurrency limit $C_{\max}$ (maximum simultaneous instances); GPU type stock availability fluctuates stochastically.

**Provenance**: the discrete-event core is adapted from SPARS (Amrizal et al., *SoftwareX* 34:102693, 2026, https://doi.org/10.1016/j.softx.2026.102693), an HPC job-scheduling simulator from Universitas Gadjah Mada. Its power-state manager and energy accounting are replaced by the CGRSP cloud fleet, stock-status provisioning model and QoS/cost metrics. SPARS is MIT-licensed (Copyright 2025 Raka Satya Prasasta); see `THIRD_PARTY_NOTICES.md`.

**Formulation**: `../thesis/chapters/ch3_formulation.tex` (Chapter 3 of the thesis)  
**Empirical data**: `../data/rendering_performance_data.csv` (6,234 real rendering jobs logged by the author's RunPod Serverless rendering pipeline: 6,030 production, 203 development-tagged, 1 untagged; the 393 jobs that ran while more than five jobs were in flight are excluded, see `../scripts/filter_over_five_concurrency.py` and `../data/unused_after_filter/`)

---

## Quick Start

```bash
cd CGRSP
# Run seven-scheduler Phase 2 comparison (hectic day ~950 jobs, C_max=5, seed 42)
python main.py

# Phase 2 normal day (100 jobs, seed 42)
python main.py --jobs 100 --arrival-rate 0.002 --output Outputs/runpod_normal --seed 42

# Constraint verifier on RH, CADR and Adaptive (seed 42 hectic day)
python verify_proposed.py

# Phase 3 statistical analysis (30 seeds, CI, paired t-tests)
python phase3_statistical.py

# Phase 3 day-type sensitivity (quiet / normal / hectic / surge)
python phase3_sensitivity.py

# Phase 3 deadline tightness sensitivity
python phase3_deadline.py

# The other Phase 3 sweeps (see the tree below) run the same way; run_all.sh runs them all

# Phase 4 monthly simulation (30 days, 10 replications)
python phase4_monthly.py
```

To regenerate everything and validate the thesis numbers in one step, run
`./run_all.sh` from the repository root (override the interpreter with
`PYTHON=python3.12 ./run_all.sh`). See `../REPRODUCE.md` for the full
reproducibility package (environment, datasets, parameters, pipeline map).

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Core simulator + FIFO baseline | Complete |
| Phase 2 | Seven-scheduler comparison (normal and hectic day, seed 42), plus constraint verification of RH, CADR and Adaptive | Complete |
| Phase 3 | Statistical analysis plus day-type, deadline, C_max, parameter, availability-model, capacity-bound and model-assumption sensitivity sweeps | Complete |
| Phase 4 | 30-day monthly simulation + publication preparation | Complete |

---

## Key Results

The authoritative, up-to-date results live in the thesis, where every
number is generated from the simulation JSON via LaTeX macros (`thesis/generated/`)
and cross-checked by `validate_numbers.py`. To avoid the
numbers here drifting out of sync, this README only summarises the findings
qualitatively:

- **SPT** achieves a lowest-tier wait (tied with RH) under capacity saturation ($\rho>1$),
  statistically tied with RH, but is significantly outperformed by EDF on
  deadline miss (n=30 seeds, paired test, survives Holm correction).
- **RH** achieves the lowest miss overall; **CADR** reaches near-RH miss at
  substantially lower tardiness; their 30-seed paired miss comparison is not
  significant, which does not establish equivalence.
- **SPT+Rescue** (laxity rescue) raises wait modestly over SPT (about 5 min at hectic saturation) while limiting
  tight-deadline starvation: the conservative fallback.
- **EDF** achieves the lowest miss among the baselines at low tight-deadline
  fractions but offers no wait improvement over FIFO.
- **FIFO/LCF/Balanced** ignore deadlines and incur the highest miss rates.
- **Concurrency ($C_{\max}$)** is the primary lever: raising it reduces both wait
  and miss under saturation.

See `thesis/chapters/ch5_experiments.tex` (Table: Summary of key experimental
findings) for the exact figures.

---

## Scheduling Algorithms

| Algorithm | File | Description |
|---|---|---|
| FIFO | `cgrsp/schedulers/fifo_scheduler.py` | Submission order baseline |
| SPT | `cgrsp/schedulers/spt_scheduler.py` | Shortest Processing Time: minimises avg wait |
| EDF | `cgrsp/schedulers/edf_scheduler.py` | Earliest Deadline First: minimises deadline misses |
| LCF | `cgrsp/schedulers/lcf_scheduler.py` | Lowest Cost First: minimises operational cost |
| Balanced | `cgrsp/schedulers/balanced_scheduler.py` | Weighted: speed (α=0.8) + cost (1−α=0.2) |
| Random | `cgrsp/schedulers/random_scheduler.py` | Random baseline |
| SPT+Rescue | `cgrsp/schedulers/laxity_rescue_scheduler.py` | SPT with laxity-based EDF rescue for near-deadline jobs |
| Adaptive | `cgrsp/schedulers/adaptive_scheduler.py` | Queue-pressure EDF-SPT hybrid with a hopeless tier |
| RH | `cgrsp/schedulers/rolling_horizon_scheduler.py` | Rolling-Horizon: forward-simulation timeline, promotes only jobs that would otherwise miss (lowest miss) |
| CADR | `cgrsp/schedulers/cost_aware_deadline_risk.py` | Cost-Aware Deadline-Risk: scale-free critical-ratio triage + cheapest deadline-feasible node (near-RH miss, much lower tardiness) |

RH and CADR are the two scheduling contributions of this work; the other eight are benchmark policies.

---

## GPU Fleet (RunPod Secure Cloud, March 2026)

| GPU | VRAM | Price ($/hr) | Empirical Avg Exec (all complexities) |
|---|---|---|---|
| RTX A5000 | 24 GB | $0.27 | 74.0 s |
| RTX 3090 | 24 GB | $0.46 | 78.1 s |
| RTX A4500 | 20 GB | $0.25 | 86.0 s |
| RTX A4000 | 16 GB | $0.25 | 97.9 s |

---

## Directory Structure

```
CGRSP/
├── cgrsp/                        # Main package
│   ├── config.py                 # ResourceType, GPU specs
│   ├── data/
│   │   ├── execution_times.py    # Empirical GPU execution time model
│   │   ├── job_generator.py      # Poisson job arrival generator
│   │   └── runpod_availability.py # Stochastic stock-status availability model
│   ├── simulator/
│   │   ├── core.py               # Discrete-event simulator
│   │   ├── fleet.py              # FleetManager base class (CTMC availability)
│   │   ├── cloud_fleet.py        # RunPodCloudFleetManager (stock-based availability)
│   │   ├── state.py              # Observable/hidden state
│   │   └── metrics.py            # Metrics collection
│   └── schedulers/
│       ├── fifo_scheduler.py
│       ├── spt_scheduler.py
│       ├── edf_scheduler.py
│       ├── lcf_scheduler.py
│       ├── balanced_scheduler.py
│       ├── random_scheduler.py
│       ├── laxity_rescue_scheduler.py  # SPT+Rescue
│       ├── adaptive_scheduler.py       # Adaptive EDF-SPT hybrid
│       ├── rolling_horizon_scheduler.py  # RH (proposed)
│       └── cost_aware_deadline_risk.py   # CADR (proposed)
├── main.py                       # CLI entry point (Phase 2 hectic and normal day)
├── verify_proposed.py            # Constraint verifier on RH, CADR, Adaptive
├── _repro.py                     # Pins PYTHONHASHSEED=0 for every entry point
├── phase3_statistical.py         # 30-seed CI + paired t-tests
├── phase3_sensitivity.py         # Day-type sensitivity sweep
├── phase3_deadline.py            # Deadline tightness sensitivity
├── phase3_beta_sensitivity.py    # Availability-model (beta_g) sensitivity
├── phase3_capacity_bound.py      # Capacity lower bound, clairvoyant relaxation
├── phase3_cmax_sensitivity.py    # C_max sensitivity
├── phase3_cr_sensitivity.py      # CADR CR* sensitivity
├── phase3_kappa_speed_sensitivity.py # Complexity-label noise and A4000 speed brackets
├── phase3_diagnostics.py         # Mean-based service, RH t_free collapse, per-seed tight misses
├── phase3_pressure_sensitivity.py # Adaptive queue-pressure sweep
├── phase3_prior_robustness.py    # Reversed-rank beta_g, RH prior ablation and RH arrival-rate prior
├── phase3_reservation_sensitivity.py # RH anticipation payoff
├── phase3_sparse_sensitivity.py  # Estimated-cell exposure and service model
├── phase3_threshold_sensitivity.py # SPT+Rescue laxity threshold
├── phase3_weight_sensitivity.py  # Objective-weight sensitivity (w_cost, alpha2/alpha1, tardiness variant)
├── phase4_monthly.py             # 30-day monthly simulation
├── Outputs/
│   ├── runpod_scheduler/         # Phase 2 hectic-day results (report.json, verification_proposed.json)
│   ├── runpod_normal/            # Phase 2 normal-day results (report.json)
│   ├── phase3/                   # Phase 3 JSON results
│   └── phase4/                   # Phase 4 monthly results
```

---

## Requirements

- Python >= 3.11
- `numpy`, `scipy`, `pandas`, `matplotlib` (see `requirements.txt`)
- `pytest` for the test suite

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## Testing

```bash
# From the CGRSP/ directory
python -m pytest -q
```

Use an interpreter that has pytest installed; the repository Makefile uses
`python3.12` (`make test` from the repository root, or
`python3.12 -m pytest -q` from this directory). The root `pyproject.toml`
sets the pytest path, so both invocations collect the full test suite.

---

## Output Files

| File | Description |
|---|---|
| `Outputs/runpod_scheduler/report.json` | Phase 2 seven-scheduler comparison, hectic day |
| `Outputs/runpod_scheduler/verification_proposed.json` | Constraint verification for RH, CADR and Adaptive |
| `Outputs/runpod_normal/report.json` | Phase 2 normal day (100 jobs), normal-day panel of the Phase 2 figure |
| `Outputs/phase3/statistical.json` | 30-seed CI and t-test results |
| `Outputs/phase3/sensitivity.json` | Day-type sensitivity |
| `Outputs/phase3/deadline.json` | Deadline tightness sensitivity |
| `Outputs/phase3/beta_sensitivity.json` | Availability-model (beta_g) sensitivity |
| `Outputs/phase3/capacity_bound.json` | Capacity lower bound, clairvoyant relaxation |
| `Outputs/phase3/cmax_sensitivity.json` | C_max sensitivity |
| `Outputs/phase3/cr_sensitivity.json` | CADR CR* sensitivity |
| `Outputs/phase3/kappa_speed_sensitivity.json` | Complexity-label noise and A4000 speed brackets |
| `Outputs/phase3/pressure_sensitivity.json` | Adaptive queue-pressure sweep |
| `Outputs/phase3/prior_robustness.json` | Reversed-rank beta_g, RH prior ablation and RH arrival-rate prior |
| `Outputs/phase3/reservation_sensitivity.json` | RH anticipation payoff |
| `Outputs/phase3/sparse_sensitivity.json` | Estimated-cell exposure and service model |
| `Outputs/phase3/threshold_sensitivity.json` | SPT+Rescue laxity threshold |
| `Outputs/phase3/weight_sensitivity.json` | Objective-weight sensitivity (w_cost, alpha2/alpha1, tardiness variant) |
| `Outputs/phase4/monthly_results.json` | Phase 4 monthly simulation (30 days, 10 reps) |
