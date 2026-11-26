"""
Phase 3: C_max Sensitivity Analysis
======================================
Tests hectic-day performance across concurrency limits C_max in {3, 5, 7, 10}
with 10 seeds each for 6 key schedulers (FIFO, SPT, EDF, SPT+Rescue, RH, CADR).

Output: Outputs/phase3/cmax_sensitivity.json
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
    CostAwareDeadlineRiskScheduler, AdaptiveScheduler,
)

SCHEDULER_MAP = {
    "FIFO":       FIFOScheduler,
    "SPT":        SPTScheduler,
    "EDF":        EDFScheduler,
    "SPT+Rescue": LaxityRescueScheduler,
    "RH":         lambda fleet_manager, seed: RollingHorizonScheduler(fleet_manager=fleet_manager, seed=seed, arrival_rate=ARRIVAL_RATE),
    "CADR":       lambda fleet_manager, seed: CostAwareDeadlineRiskScheduler(fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=True),
    "Adaptive":   lambda fleet_manager, seed: AdaptiveScheduler(fleet_manager, seed=seed),
}

# Baseline 5 is RunPod's default quota. Override the sweep with e.g.
# CMAX_VALUES="3,5,8,12" python3 phase3_cmax_sensitivity.py.
import os
CMAX_VALUES = [int(c) for c in os.environ.get("CMAX_VALUES", "3,5,7,10").split(",")]
N_SEEDS = 10
N_JOBS = 950
ARRIVAL_RATE = 0.100
DEADLINE_DIST = [(3600.0, 0.20), (28800.0, 0.80)]
DURATION = 86400.0


def _run_one(scheduler_cls, seed, cmax):
    job_config = JobGeneratorConfig(
        num_jobs=N_JOBS, arrival_rate=ARRIVAL_RATE, seed=seed,
        deadline_distribution=DEADLINE_DIST,
    )
    jobs = JobGenerator(job_config).generate()
    cloud = RunPodCloudFleetManager(
        max_concurrency=cmax,
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
    m = sim.get_metrics_summary()
    return {
        "avg_wait_min": round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "deadline_miss_pct": round(m["job_counts"]["deadline_miss_rate_pct"], 4),
    }


def main():
    out_path = Path(__file__).parent / "Outputs/phase3/cmax_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()

    for cmax in CMAX_VALUES:
        key = f"cmax{cmax}"
        print(f"\nC_max={cmax}:")
        sched_results = {}
        for sname, scls in SCHEDULER_MAP.items():
            runs = []
            for seed in range(N_SEEDS):
                runs.append(_run_one(scls, seed, cmax))
                print(f"  {sname} seed={seed} done", flush=True)
            sched_results[sname] = {
                "avg_wait_min": round(float(np.mean([r["avg_wait_min"] for r in runs])), 4),
                "avg_miss_pct": round(float(np.mean([r["deadline_miss_pct"] for r in runs])), 4),
                "std_wait_min": round(float(np.std([r["avg_wait_min"] for r in runs], ddof=1)), 4),
                "std_miss_pct": round(float(np.std([r["deadline_miss_pct"] for r in runs], ddof=1)), 4),
            }
        results[key] = {"cmax": cmax, "n_seeds": N_SEEDS, "schedulers": sched_results}

    output = {
        "metadata": {
            "n_jobs": N_JOBS, "arrival_rate": ARRIVAL_RATE,
            "cmax_values": CMAX_VALUES, "n_seeds": N_SEEDS,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")

    print("\nSummary (miss %):")
    header = f"{'Scheduler':<12}" + "".join(f"  C={c}" for c in CMAX_VALUES)
    print(header)
    for sname in SCHEDULER_MAP:
        row = f"{sname:<12}"
        for cmax in CMAX_VALUES:
            v = results[f"cmax{cmax}"]["schedulers"][sname]["avg_miss_pct"]
            row += f"  {v:5.1f}"
        print(row)


if __name__ == "__main__":
    main()
