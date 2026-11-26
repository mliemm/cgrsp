"""
Phase 3 — SPT+Rescue Threshold Sensitivity
============================================
Sweeps the laxity rescue threshold theta in {60, 180, 300, 600, 1200} seconds
on hectic-day workload with 10 seeds. SPT+Rescue only.

Output: Outputs/phase3/threshold_sensitivity.json
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
from cgrsp.schedulers.laxity_rescue_scheduler import LaxityRescueScheduler

THETA_VALUES = [60, 180, 300, 600, 1200]
N_SEEDS = 10
N_JOBS = 950
ARRIVAL_RATE = 0.100
DEADLINE_DIST = [(3600.0, 0.20), (28800.0, 0.80)]
DURATION = 86400.0
CONCURRENCY = 5


def _run_one(seed, theta):
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
    sched = LaxityRescueScheduler(fleet_manager=cloud, seed=seed, rescue_threshold=theta)
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
    out_path = Path(__file__).parent / "Outputs/phase3/threshold_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()

    for theta in THETA_VALUES:
        key = f"theta{theta}"
        print(f"\ntheta={theta}s:", flush=True)
        runs = [_run_one(seed, theta) for seed in range(N_SEEDS)]
        results[key] = {
            "theta_s": theta,
            "n_seeds": N_SEEDS,
            "avg_wait_min": round(float(np.mean([r["avg_wait_min"] for r in runs])), 4),
            "avg_miss_pct": round(float(np.mean([r["deadline_miss_pct"] for r in runs])), 4),
            "std_wait_min": round(float(np.std([r["avg_wait_min"] for r in runs], ddof=1)), 4),
            "std_miss_pct": round(float(np.std([r["deadline_miss_pct"] for r in runs], ddof=1)), 4),
        }
        print(f"  wait={results[key]['avg_wait_min']:.2f} min  miss={results[key]['avg_miss_pct']:.2f}%")

    output = {
        "metadata": {
            "n_jobs": N_JOBS, "arrival_rate": ARRIVAL_RATE,
            "theta_values": THETA_VALUES, "n_seeds": N_SEEDS,
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
