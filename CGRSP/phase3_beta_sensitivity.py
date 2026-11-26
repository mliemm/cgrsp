"""
Phase 3 — Availability-Model (β_g) Sensitivity Analysis
========================================================
Sweeps the baseline high-stock probabilities β_g of the availability model
by a common scale factor and re-runs the hectic-day benchmark, testing the
robustness of the scheduler comparison to the availability-model assumption
(the β_g values are modelling assumptions, not measured frequencies).

Scale 1.00 is the calibrated baseline. Lower scales shrink the High-stock
probability of every type, pushing dispatches into Medium (30-120 s delay)
and, in the peak band, Low (600-7200 s delay).

Hectic day (950 jobs, λ=0.100 j/s), C_max=5, 30 seeds, same seeds and job
streams as the Phase 3 statistical benchmark.

Output: Outputs/phase3/beta_sensitivity.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys, os, json, time
from pathlib import Path

import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.data.runpod_availability import RunPodAvailabilityModel
from cgrsp.schedulers import (
    FIFOScheduler, SPTScheduler, EDFScheduler, LCFScheduler,
    BalancedScheduler, RandomScheduler, LaxityRescueScheduler,
    RollingHorizonScheduler, CostAwareDeadlineRiskScheduler, AdaptiveScheduler,
)

# All ten policies are swept; the thesis table shows the six that define the
# headline ordering, the JSON carries the full grid.
SCHEDULER_MAP = {
    "FIFO":       lambda fm, seed, ar: FIFOScheduler(fm, seed=seed),
    "SPT":        lambda fm, seed, ar: SPTScheduler(fm, seed=seed),
    "EDF":        lambda fm, seed, ar: EDFScheduler(fm, seed=seed),
    "LCF":        lambda fm, seed, ar: LCFScheduler(fm, seed=seed),
    "Balanced":   lambda fm, seed, ar: BalancedScheduler(fm, seed=seed),
    "Random":     lambda fm, seed, ar: RandomScheduler(fm, seed=seed),
    "SPT+Rescue": lambda fm, seed, ar: LaxityRescueScheduler(fm, seed=seed),
    "RH":         lambda fm, seed, ar: RollingHorizonScheduler(
        fleet_manager=fm, seed=seed, arrival_rate=ar),
    "CADR":       lambda fm, seed, ar: CostAwareDeadlineRiskScheduler(
        fleet_manager=fm, seed=seed, cr_star=3.0, cost_aware=True),
    "Adaptive":   lambda fm, seed, ar: AdaptiveScheduler(fm, seed=seed),
}

CONCURRENCY = 5
N_SEEDS     = 30
DURATION    = 86400.0
N_JOBS      = 950
ARRIVAL     = 0.100
DEADLINES   = [(3600.0, 0.20), (28800.0, 0.80)]
SCALES      = [0.25, 0.50, 0.75, 1.00, 1.25]


def _run_one(sched_name, seed, scale):
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
    # Shadow the class-level table with a scaled instance copy. The model
    # reads self.BASE_HIGH_STOCK_PROB, so this affects only this run; the
    # p_high = min(0.95, beta*m(t)) cap still applies downstream.
    cloud.availability_model.BASE_HIGH_STOCK_PROB = {
        k: v * scale
        for k, v in RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB.items()
    }

    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sim.set_scheduler(SCHEDULER_MAP[sched_name](cloud, seed, ARRIVAL))

    for job in jobs:
        sim.schedule_event(
            EventType.JOB_ARRIVAL, job.submit_time,
            {"complexity": job.complexity, "deadline": job.deadline,
             "estimated_duration": job.estimated_duration or 100.0},
        )
    sim.run(duration=DURATION)
    m = sim.get_metrics_summary()
    return {
        "wait_min": m["timing_metrics"]["avg_waiting_time"] / 60.0,
        "miss_pct": m["job_counts"]["deadline_miss_rate_pct"],
        "tard_min": float(m.get("avg_tardiness_s", 0.0)) / 60.0,
        "completed": m["job_counts"]["completed"],
    }


def _run_one_packed(args):
    """Pool.map shim: unpack a (scheduler, seed, scale) tuple."""
    return _run_one(*args)


def main():
    t0 = time.time()
    results = {}
    # Each (scheduler, seed, scale) cell builds its own seeded job stream, fleet
    # and simulator and shares no mutable state, so the grid is order
    # independent and a process pool reproduces the sequential result exactly.
    # Sequentially this sweep is the pipeline's bottleneck at about six hours.
    grid = [(name, seed, scale)
            for scale in SCALES for name in SCHEDULER_MAP for seed in range(N_SEEDS)]
    with Pool(processes=min(cpu_count(), 8)) as pool:
        flat = pool.map(_run_one_packed, grid)
    cells = dict(zip(grid, flat))

    for scale in SCALES:
        results[f"{scale:.2f}"] = {}
        for name in SCHEDULER_MAP:
            waits, misses, tards, comps = [], [], [], []
            for seed in range(N_SEEDS):
                r = cells[(name, seed, scale)]
                waits.append(r["wait_min"])
                misses.append(r["miss_pct"])
                tards.append(r["tard_min"])
                comps.append(r["completed"])
            # effective miss per seed: a job unfinished at the end of the
            # 24 h window has certainly missed (all deadlines fall before the
            # window end), so on-time jobs = completed * (1 - miss/100)
            eff = [100.0 - (c / N_JOBS) * (100.0 - m) for c, m in zip(comps, misses)]
            results[f"{scale:.2f}"][name] = {
                "wait_min_mean": round(float(np.mean(waits)), 4),
                "miss_pct_mean": round(float(np.mean(misses)), 4),
                "tard_min_mean": round(float(np.mean(tards)), 4),
                "completed_mean": round(float(np.mean(comps)), 2),
                "completed_pct_mean": round(100.0 * float(np.mean(comps)) / N_JOBS, 2),
                "eff_miss_pct_mean": round(float(np.mean(eff)), 4),
                "miss_pct_raw": [round(x, 4) for x in misses],
                "completed_raw": comps,
                "eff_miss_pct_raw": [round(x, 4) for x in eff],
            }
            print(f"scale {scale:.2f} {name:11s} "
                  f"miss {results[f'{scale:.2f}'][name]['miss_pct_mean']:6.2f}%  "
                  f"wait {results[f'{scale:.2f}'][name]['wait_min_mean']:7.2f} min",
                  flush=True)

    out = {
        "description": "Availability-model beta_g scale sweep, hectic day, "
                       "C_max=5, 30 seeds",
        "scales": [f"{x:.2f}" for x in SCALES],  # string keys matching results
        "n_seeds": N_SEEDS,
        "results": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = Path(__file__).parent / "Outputs" / "phase3" / "beta_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
