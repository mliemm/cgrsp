"""
Phase 3 — Adaptive Queue-Pressure Threshold (P) Sensitivity
============================================================
Sweeps the Adaptive scheduler's queue-pressure threshold P (the queue length
above which the critical tier widens to the full loose window), the one tier
threshold in the scheduler family not previously swept. Same protocol as the
rescue-threshold and CR* sweeps: hectic day, C_max=5, n=10 seeds.

Output: Outputs/phase3/pressure_sensitivity.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 and TZ=Asia/Taipei

import sys, os, json, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers import AdaptiveScheduler

CONCURRENCY = 5
N_SEEDS     = 10
DURATION    = 86400.0
N_JOBS      = 950
ARRIVAL     = 0.100
DEADLINES   = [(3600.0, 0.20), (28800.0, 0.80)]
P_VALUES    = [3, 5, 10, 20, 50]


def _run_one(seed, p_value):
    job_config = JobGeneratorConfig(
        num_jobs=N_JOBS, arrival_rate=ARRIVAL, seed=seed,
        deadline_distribution=DEADLINES,
    )
    jobs = JobGenerator(job_config).generate()
    cloud = RunPodCloudFleetManager(
        max_concurrency=CONCURRENCY,
        supported_types=ResourceType.supported_gpus(),
        random_seed=seed,
    )
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sim.set_scheduler(AdaptiveScheduler(cloud, seed=seed,
                                        pressure_threshold=p_value))
    for job in jobs:
        sim.schedule_event(
            EventType.JOB_ARRIVAL, job.submit_time,
            {"complexity": job.complexity, "deadline": job.deadline,
             "estimated_duration": job.estimated_duration or 100.0},
        )
    sim.run(duration=DURATION)
    m = sim.get_metrics_summary()
    return (m["timing_metrics"]["avg_waiting_time"] / 60.0,
            m["job_counts"]["deadline_miss_rate_pct"],
            float(m.get("avg_tardiness_s", 0.0)) / 60.0)


def main():
    t0 = time.time()
    results = {}
    for p in P_VALUES:
        waits, misses, tards = [], [], []
        for seed in range(N_SEEDS):
            w, mi, ta = _run_one(seed, p)
            waits.append(w); misses.append(mi); tards.append(ta)
        results[str(p)] = {
            "wait_min_mean": round(float(np.mean(waits)), 4),
            "miss_pct_mean": round(float(np.mean(misses)), 4),
            "tard_min_mean": round(float(np.mean(tards)), 4),
            "miss_pct_raw": [round(x, 4) for x in misses],
        }
        c = results[str(p)]
        print(f"P={p:3d}  miss {c['miss_pct_mean']:6.2f}%  "
              f"wait {c['wait_min_mean']:7.2f} min  "
              f"tard {c['tard_min_mean']:6.2f} min", flush=True)

    out = {
        "description": "Adaptive queue-pressure threshold sweep, hectic day, "
                       "C_max=5, n=10 seeds (protocol matches theta and CR* sweeps)",
        "p_values": P_VALUES,
        "n_seeds": N_SEEDS,
        "results": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = Path(__file__).parent / "Outputs" / "phase3" / "pressure_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
