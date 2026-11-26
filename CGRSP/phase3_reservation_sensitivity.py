"""
Phase 3 -- Rolling-Horizon Anticipation (Reservation) Sensitivity
=================================================================
Demonstrates the value of load-gated tight-deadline reservation in the
Rolling-Horizon scheduler.

Two day types are swept over the reservation count rho_res in {0, 1, 2, 3}:

  * surge  (lambda=0.020, below saturation): reservation holds idle capacity for
    anticipated tight (rush) arrivals. Expected to cut tight-deadline miss
    substantially at a modest wait cost.
  * hectic (lambda=0.100, above saturation): the offered-load gate disables
    reservation, so every rho_res reproduces the no-reservation result. This
    shows the mechanism correctly switches itself off when there is no spare
    capacity to reserve.

Per setting we report overall wait/miss plus tight-deadline and loose-deadline
miss separately (anticipation targets the tight, high-value jobs).

Output: Outputs/phase3/reservation_sensitivity.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys
import os
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers.rolling_horizon_scheduler import RollingHorizonScheduler

RHO_RES_VALUES = [0, 1, 2, 3]
N_SEEDS = 30
CONCURRENCY = 5
DURATION = 86400.0
DEADLINE_DIST = [(3600.0, 0.20), (28800.0, 0.80)]
TIGHT_WINDOW = 7200.0

DAYS = {
    "surge":  {"n_jobs": 730, "arrival_rate": 0.020},
    "hectic": {"n_jobs": 950, "arrival_rate": 0.100},
}


def _run_one(seed, rho_res, n_jobs, arrival_rate):
    jc = JobGeneratorConfig(num_jobs=n_jobs, arrival_rate=arrival_rate, seed=seed,
                            deadline_distribution=DEADLINE_DIST)
    jobs = JobGenerator(jc).generate()
    cloud = RunPodCloudFleetManager(max_concurrency=CONCURRENCY,
                                    supported_types=ResourceType.supported_gpus(),
                                    random_seed=seed)
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sched = RollingHorizonScheduler(fleet_manager=cloud, seed=seed,
                                    reservation_aggressiveness=float(rho_res),
                                    arrival_rate=arrival_rate)
    sim.set_scheduler(sched)
    for j in jobs:
        sim.schedule_event(EventType.JOB_ARRIVAL, j.submit_time,
                           {"complexity": j.complexity, "deadline": j.deadline,
                            "estimated_duration": j.estimated_duration or 100.0})
    sim.run(duration=DURATION)
    jm = sim.metrics_collector._job_metrics
    tn = tm = ln = lm = 0
    for r in jm.values():
        if r.completion_time is None:
            continue
        if (r.deadline - r.submit_time) < TIGHT_WINDOW:
            tn += 1
            tm += 0 if r.deadline_met else 1
        else:
            ln += 1
            lm += 0 if r.deadline_met else 1
    m = sim.get_metrics_summary()
    return {
        "avg_wait_min": m["timing_metrics"]["avg_waiting_time"] / 60.0,
        "deadline_miss_pct": m["job_counts"]["deadline_miss_rate_pct"],
        "tight_miss_pct": 100.0 * tm / tn if tn else 0.0,
        "loose_miss_pct": 100.0 * lm / ln if ln else 0.0,
        "avg_tardiness_min": m.get("avg_tardiness_s", 0.0) / 60.0,
    }


def main():
    out_path = Path(__file__).parent / "Outputs/phase3/reservation_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for day, p in DAYS.items():
        results[day] = {}
        print(f"\n[{day}] lambda={p['arrival_rate']}, n_jobs={p['n_jobs']}", flush=True)
        for rho in RHO_RES_VALUES:
            runs = [_run_one(seed, rho, p["n_jobs"], p["arrival_rate"]) for seed in range(N_SEEDS)]
            agg = {
                "rho_res": rho,
                "n_seeds": N_SEEDS,
                "avg_wait_min": round(float(np.mean([r["avg_wait_min"] for r in runs])), 4),
                "avg_miss_pct": round(float(np.mean([r["deadline_miss_pct"] for r in runs])), 4),
                "tight_miss_pct": round(float(np.mean([r["tight_miss_pct"] for r in runs])), 4),
                "loose_miss_pct": round(float(np.mean([r["loose_miss_pct"] for r in runs])), 4),
                "avg_tardiness_min": round(float(np.mean([r["avg_tardiness_min"] for r in runs])), 4),
            }
            results[day][f"r{rho}"] = agg
            print(f"  rho_res={rho}: wait={agg['avg_wait_min']:.2f}  miss={agg['avg_miss_pct']:.2f}%  "
                  f"tight={agg['tight_miss_pct']:.2f}%  loose={agg['loose_miss_pct']:.2f}%", flush=True)

    output = {
        "metadata": {
            "rho_res_values": RHO_RES_VALUES, "n_seeds": N_SEEDS,
            "concurrency": CONCURRENCY, "days": DAYS,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
