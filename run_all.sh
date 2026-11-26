#!/bin/bash
# Regenerate ALL CGRSP simulation outputs, then rebuild the thesis data
# fragments and figures, and validate every reported number end to end.
#
# Determinism: each experiment entry point imports CGRSP/_repro.py, which pins
# PYTHONHASHSEED=0; combined with the crc32-based provisioning seed this makes
# every re-run byte-for-byte identical. Only the proposed scheduler's code
# should ever change results.
#
# Override the interpreter with e.g. PYTHON=python3.12 ./run_all.sh
set -euo pipefail

PY="${PYTHON:-python3}"
export PYTHONPATH=CGRSP

echo "==> Phase 2 — hectic day (950 jobs, C_max=5, lambda=0.100, seed 42)..."
$PY CGRSP/main.py --output CGRSP/Outputs/runpod_scheduler > phase2.log
$PY CGRSP/verify_proposed.py > phase2_verify_proposed.log

echo "==> Phase 2: normal day (100 jobs, lambda=0.002, seed 42; normal-day panel of the Phase 2 figure)..."
$PY CGRSP/main.py --jobs 100 --arrival-rate 0.002 --output CGRSP/Outputs/runpod_normal --seed 42 > phase2_normal.log

echo "==> Phase 3 — statistical analysis (30 seeds)..."
$PY CGRSP/phase3_statistical.py --seeds 30 > phase3_stat.log

echo "==> Phase 3 — day-type sensitivity..."
$PY CGRSP/phase3_sensitivity.py > phase3_sensitivity.log

echo "==> Phase 3 — deadline-tightness sweep..."
$PY CGRSP/phase3_deadline.py > phase3_deadline.log

echo "==> Phase 3 — C_max sensitivity (3/5/7/10)..."
$PY CGRSP/phase3_cmax_sensitivity.py > phase3_cmax.log

echo "==> Phase 3 — rescue-threshold sensitivity..."
$PY CGRSP/phase3_threshold_sensitivity.py > phase3_threshold.log

echo "==> Phase 3 — CADR critical-ratio sensitivity..."
$PY CGRSP/phase3_cr_sensitivity.py > phase3_cr.log

echo "==> Phase 3 — RH anticipation reservation sensitivity..."
$PY CGRSP/phase3_reservation_sensitivity.py > phase3_reservation.log

echo "==> Phase 3 — availability-model (beta) sensitivity..."
$PY CGRSP/phase3_beta_sensitivity.py > phase3_beta.log

echo "==> Phase 3 — capacity lower bound..."
$PY CGRSP/phase3_capacity_bound.py > phase3_capbound.log

echo "==> Phase 3 — availability-prior robustness (rank reversal + RH prior ablation)..."
$PY CGRSP/phase3_prior_robustness.py > phase3_prior.log

echo "==> Phase 3 — Adaptive queue-pressure threshold sensitivity..."
$PY CGRSP/phase3_pressure_sensitivity.py > phase3_pressure.log

echo "==> Phase 3 — tardiness-objective weight sensitivity..."
$PY CGRSP/phase3_weight_sensitivity.py > phase3_weight.log

echo "==> Phase 3 estimated-cell mean and dispersion sensitivity..."
$PY CGRSP/phase3_sparse_sensitivity.py > phase3_sparse.log

echo "==> Phase 3 complexity-label noise and A4000 speed sensitivity..."
$PY CGRSP/phase3_kappa_speed_sensitivity.py > phase3_kappa_speed.log

echo "==> Phase 3 diagnostics (mean-based service, RH t_free collapse, per-seed tight misses)..."
$PY CGRSP/phase3_diagnostics.py > phase3_diagnostics.log

echo "==> Phase 3 — classical and learned baselines (30 seeds; about 4 min)..."
$PY CGRSP/phase3_baselines.py > phase3_baselines.log

echo "==> Phase 3 — fleet-composition sensitivity (six mixes, 10 seeds; about 7 min)..."
$PY CGRSP/phase3_fleet_mix.py > phase3_fleet_mix.log

echo "==> Phase 4 — monthly simulation (30 days, 10 replications)..."
$PY CGRSP/phase4_monthly.py > phase4_monthly.log

echo "==> Regenerating thesis data fragments (tables + stats macros)..."
$PY generate_data.py

echo "==> Regenerating figures..."
$PY thesis/figures/generate_figures.py
$PY generate_gantt.py

echo "==> Validating all reported numbers, conventions, and spelling..."
$PY validate_numbers.py

echo "==> Building the thesis PDF and updating public/thesis.pdf..."
"$(dirname "$0")/scripts/build_thesis.sh"

echo "Done. Thesis and public PDFs match."
