"""
Phase 3 - Objective-Weight Sensitivity Analysis
================================================
Tests whether the qualitative scheduler ranking on the bi-objective score is
robust to the choice of objective-function weights. The decomposed metrics
(wait, miss, cost) are weight-independent by construction (no scheduler optimises
the scalar objective directly); this experiment sweeps the weights post-hoc over
the per-seed totals to show how the composite ranking responds.

Two sweeps on the hectic-day workload (n=10 seeds, C_max=5):
  A. Multi-objective weight w_cost in {0.1, 0.3, 0.5, 0.7, 0.9} (w_qos = 1 - w_cost),
     with the QoS-internal ratio fixed at the baseline (alpha1=1, alpha2=10).
  B. Deadline-miss penalty ratio alpha2/alpha1 in {1, 5, 10, 20, 50} (alpha1=1),
     with w_qos = w_cost = 0.5.

The QoS component is mean waiting time in seconds plus alpha2 times the
missed-job fraction. Thus alpha2/alpha1 is seconds of wait per missed job,
matching the mathematical model. The tardiness variant uses seconds for both
wait and tardiness.

Because the QoS component and cost component (USD) are on
different scales, each component is min-max normalised across the six schedulers
before the weights are applied, so that w_cost has a meaningful effect.

Output: Outputs/phase3/weight_sensitivity.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys, os, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers import (
    FIFOScheduler, SPTScheduler, EDFScheduler,
    LaxityRescueScheduler, RollingHorizonScheduler,
    CostAwareDeadlineRiskScheduler,
)

SCHEDULER_MAP = {
    "FIFO":       FIFOScheduler,
    "SPT":        SPTScheduler,
    "EDF":        EDFScheduler,
    "SPT+Rescue": LaxityRescueScheduler,
    "RH":         lambda fleet_manager, seed: RollingHorizonScheduler(fleet_manager=fleet_manager, seed=seed, arrival_rate=ARRIVAL_RATE),
    "CADR":       lambda fleet_manager, seed: CostAwareDeadlineRiskScheduler(fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=True),
}

N_SEEDS = 10
N_JOBS = 950
ARRIVAL_RATE = 0.100
DEADLINE_DIST = [(3600.0, 0.20), (28800.0, 0.80)]
DURATION = 86400.0
CONCURRENCY = 5

# Sweep grids
WCOST_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]      # multi-objective weight (w_qos = 1 - w_cost)
ALPHA2_VALUES = [1, 5, 10, 20, 50]            # deadline-penalty ratio alpha2/alpha1 (alpha1 = 1)
BASE_ALPHA1, BASE_ALPHA2 = 1.0, 10.0


def _run_one(scheduler_cls, seed):
    job_config = JobGeneratorConfig(
        num_jobs=N_JOBS, arrival_rate=ARRIVAL_RATE, seed=seed,
        deadline_distribution=DEADLINE_DIST,
    )
    jobs = JobGenerator(job_config).generate()
    cloud = RunPodCloudFleetManager(
        max_concurrency=CONCURRENCY,
        supported_types=ResourceType.supported_gpus(),
        random_seed=seed,
    )
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sched = scheduler_cls(fleet_manager=cloud, seed=seed)
    sim.set_scheduler(sched)
    for job in jobs:
        sim.schedule_event(
            EventType.JOB_ARRIVAL, job.submit_time,
            {"complexity": job.complexity, "deadline": job.deadline,
             "estimated_duration": job.estimated_duration or 100.0},
        )
    sim.run(duration=DURATION)
    sm = sim.metrics_collector.system_metrics
    assert sm.total_jobs_completed == N_JOBS, "Weight scoring requires the complete job cohort"
    m = sim.get_metrics_summary()
    return {
        "completed":         sm.total_jobs_completed,
        "missed":            sm.total_jobs_deadline_missed,
        "total_wait_s":      float(sm.total_waiting_time),
        "total_tard_s":      float(sm.total_tardiness),
        "total_cost":        float(sm.total_execution_cost),
        "avg_wait_min":      round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "deadline_miss_pct": round(m["job_counts"]["deadline_miss_rate_pct"], 4),
        "avg_tardiness_min": round(m["avg_tardiness_s"] / 60.0, 4),
    }


def _minmax(values):
    """Min-max normalise to [0,1]; 0 = best (lowest), 1 = worst. Flat -> all 0."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def qos_score(wait_seconds, missed_fraction, alpha1=1.0, alpha2=10.0):
    """Mean model penalty, with seconds of wait and a per-job miss indicator."""
    return alpha1 * wait_seconds + alpha2 * missed_fraction


def main():
    out_path = Path(__file__).parent / "Outputs/phase3/weight_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Collect per-seed totals for each scheduler
    raw = {}
    for sname, scls in SCHEDULER_MAP.items():
        runs = [_run_one(scls, seed) for seed in range(N_SEEDS)]
        raw[sname] = {
            "mean_wait_s": float(np.mean([r["total_wait_s"] / r["completed"] for r in runs])),
            "mean_miss_fraction": float(np.mean([r["missed"] / r["completed"] for r in runs])),
            "mean_total_wait_s": float(np.mean([r["total_wait_s"] for r in runs])),
            "mean_total_tard_s": float(np.mean([r["total_tard_s"] for r in runs])),
            "mean_total_cost":   float(np.mean([r["total_cost"] for r in runs])),
            "avg_wait_min":      round(float(np.mean([r["avg_wait_min"] for r in runs])), 4),
            "miss_pct":          round(float(np.mean([r["deadline_miss_pct"] for r in runs])), 4),
            "mean_tard_min":     round(float(np.mean([r["avg_tardiness_min"] for r in runs])), 4),
        }
        print(f"  {sname:<11} wait={raw[sname]['avg_wait_min']:.2f}min "
              f"miss={raw[sname]['miss_pct']:.2f}% cost=${raw[sname]['mean_total_cost']:.2f}", flush=True)

    scheds = list(SCHEDULER_MAP.keys())
    cost_norm = _minmax([raw[s]["mean_total_cost"] for s in scheds])
    cost_norm = {s: cost_norm[i] for i, s in enumerate(scheds)}

    def qos_norm_for(alpha1, alpha2):
        qos_raw = [qos_score(raw[s]["mean_wait_s"], raw[s]["mean_miss_fraction"], alpha1, alpha2)
                   for s in scheds]
        nq = _minmax(qos_raw)
        return {s: nq[i] for i, s in enumerate(scheds)}

    # 2a. Sweep A: multi-objective weight w_cost (QoS ratio fixed at baseline)
    qn_base = qos_norm_for(BASE_ALPHA1, BASE_ALPHA2)
    sweep_wcost = {}
    for wc in WCOST_VALUES:
        wq = 1.0 - wc
        comp = {s: round(wq * qn_base[s] + wc * cost_norm[s], 4) for s in scheds}
        winner = min(comp, key=comp.get)
        sweep_wcost[f"{wc:.1f}"] = {"w_qos": round(wq, 1), "w_cost": wc,
                                    "composite": comp, "winner": winner}

    # 2b. Sweep B: deadline-penalty ratio alpha2/alpha1 (w_qos = w_cost = 0.5)
    sweep_alpha2 = {}
    for a2 in ALPHA2_VALUES:
        qn = qos_norm_for(BASE_ALPHA1, float(a2))
        comp = {s: round(0.5 * qn[s] + 0.5 * cost_norm[s], 4) for s in scheds}
        winner = min(comp, key=comp.get)
        sweep_alpha2[str(a2)] = {"alpha2_over_alpha1": a2, "composite": comp, "winner": winner}

    # 2c. Tardiness objective: parallel alpha2/alpha1 sweep using normalised mean tardiness
    #     instead of normalised miss_pct as the QoS component (w_qos = w_cost = 0.5).
    def tard_qos_norm_for(alpha1, alpha2):
        qos_raw = [alpha1 * raw[s]["mean_wait_s"] + alpha2 * raw[s]["mean_total_tard_s"] / N_JOBS
                   for s in scheds]
        nq = _minmax(qos_raw)
        return {s: nq[i] for i, s in enumerate(scheds)}

    tardiness_objective = {}
    for a2 in ALPHA2_VALUES:
        tqn = tard_qos_norm_for(BASE_ALPHA1, float(a2))
        comp = {s: round(0.5 * tqn[s] + 0.5 * cost_norm[s], 4) for s in scheds}
        winner = min(comp, key=comp.get)
        tardiness_objective[str(a2)] = {"alpha2_over_alpha1": a2, "composite": comp, "winner": winner}

    output = {
        "metadata": {
            "n_jobs": N_JOBS, "arrival_rate": ARRIVAL_RATE, "n_seeds": N_SEEDS,
            "concurrency": CONCURRENCY, "wcost_values": WCOST_VALUES,
            "alpha2_values": ALPHA2_VALUES, "base_alpha1": BASE_ALPHA1,
            "base_alpha2": BASE_ALPHA2,
            "note": "QoS uses seconds and missed-job fractions; components are min-max normalised across six schedulers before weighting.",
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "raw": raw,
        "sweep_wcost": sweep_wcost,
        "sweep_alpha2": sweep_alpha2,
        "tardiness_objective": tardiness_objective,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")

    print("\nSweep A (w_cost): winner per setting")
    for wc, r in sweep_wcost.items():
        print(f"  w_cost={wc}: {r['winner']}")
    print("Sweep B (alpha2/alpha1): winner per setting")
    for a2, r in sweep_alpha2.items():
        print(f"  alpha2={a2}: {r['winner']}")
    print("Tardiness objective (alpha2/alpha1): winner per setting")
    for a2, r in tardiness_objective.items():
        print(f"  alpha2={a2}: {r['winner']}")


if __name__ == "__main__":
    main()
