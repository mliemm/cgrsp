"""
Phase 3 — Day-Type Sensitivity Analysis
=========================================
Compares all schedulers across four day types
(quiet / normal / hectic / surge) at C_max=5.  Replaces the abstract
arrival-rate sweep with operationally grounded workload scenarios derived
from two empirical sources:

  Source A — rendering dataset (6,234 jobs, July 2025 to June 2026)
  Source B — RunPod billing explorer Apr–May 2026 (30 complete days)

The hectic and surge job counts are assumed stress loads of the order of the
busiest billing days, not statistics of the record.

Each day type is run for a full 24-hour window (86 400 s), 30 seeds.

Output: Outputs/phase3/sensitivity.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys, os, json, math, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers import (
    FIFOScheduler, SPTScheduler, EDFScheduler,
    LCFScheduler, BalancedScheduler, RandomScheduler, LaxityRescueScheduler,
    RollingHorizonScheduler, CostAwareDeadlineRiskScheduler, AdaptiveScheduler,
)

SCHEDULER_MAP = {
    # Each factory takes (fleet_manager, seed, arrival_rate); only RH uses the
    # arrival rate (its load gate needs the day's true offered load).
    "FIFO":       lambda fleet_manager, seed, ar: FIFOScheduler(fleet_manager, seed=seed),
    "SPT":        lambda fleet_manager, seed, ar: SPTScheduler(fleet_manager, seed=seed),
    "EDF":        lambda fleet_manager, seed, ar: EDFScheduler(fleet_manager, seed=seed),
    "LCF":        lambda fleet_manager, seed, ar: LCFScheduler(fleet_manager, seed=seed),
    "Balanced":   lambda fleet_manager, seed, ar: BalancedScheduler(fleet_manager, seed=seed),
    "Random":     lambda fleet_manager, seed, ar: RandomScheduler(fleet_manager, seed=seed),
    "SPT+Rescue": lambda fleet_manager, seed, ar: LaxityRescueScheduler(fleet_manager, seed=seed),
    "RH":         lambda fleet_manager, seed, ar: RollingHorizonScheduler(
        fleet_manager=fleet_manager, seed=seed, arrival_rate=ar),
    "CADR":       lambda fleet_manager, seed, ar: CostAwareDeadlineRiskScheduler(
        fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=True),
    "Adaptive":   lambda fleet_manager, seed, ar: AdaptiveScheduler(fleet_manager, seed=seed),
}

CONCURRENCY = 5       # concurrency budget; equals RunPod default per-account worker quota
N_SEEDS     = 30
DURATION    = 86400.0

# Day-type parameters: quiet and normal follow the billing record; hectic and surge are assumed stress loads
DAY_TYPE_PARAMS = {
    "quiet":  {"n_jobs": 6,   "arrival_rate": 0.0008,
               "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)]},
    "normal": {"n_jobs": 100, "arrival_rate": 0.002,
               "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)]},
    "hectic": {"n_jobs": 950, "arrival_rate": 0.100,
               "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)]},
    "surge":  {"n_jobs": 730, "arrival_rate": 0.020,
               "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)]},
}
DAY_TYPES = ["quiet", "normal", "hectic", "surge"]


# ─── simulation runner ───────────────────────────────────────────────────────

def _run_one(scheduler_cls, seed, day_type):
    p = DAY_TYPE_PARAMS[day_type]
    job_config = JobGeneratorConfig(
        num_jobs=p["n_jobs"],
        arrival_rate=p["arrival_rate"],
        seed=seed,
        deadline_distribution=p["deadline_dist"],
    )
    jobs = JobGenerator(job_config).generate()

    cloud = RunPodCloudFleetManager(
        max_concurrency=CONCURRENCY,
        supported_types=ResourceType.supported_gpus(),
        random_seed=seed,
    )
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sched = scheduler_cls(cloud, seed, p["arrival_rate"])
    sim.set_scheduler(sched)

    for job in jobs:
        sim.schedule_event(
            EventType.JOB_ARRIVAL,
            job.submit_time,
            {
                "complexity": job.complexity,
                "deadline": job.deadline,
                "estimated_duration": job.estimated_duration or 100.0,
            },
        )

    sim.run(duration=DURATION)
    m = sim.get_metrics_summary()
    return {
        "avg_wait_min":      round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "avg_miss_pct":      round(m["job_counts"]["deadline_miss_rate_pct"], 4),
        "total_cost":        round(m["cost_metrics"]["total_cost"], 6),
        "utilization_pct":   round(m["utilization"]["avg_pct"], 4),
        "completed":         m["job_counts"]["completed"],
        "avg_tardiness_min": m.get("avg_tardiness_s", 0.0) / 60.0,
        "p95_tardiness_min": m.get("p95_tardiness_s", 0.0) / 60.0,
    }


def _mean(vals):
    return round(float(np.mean(vals)), 4) if vals else None


# ─── main ────────────────────────────────────────────────────────────────────

def run_sensitivity_analysis(verbose: bool = True) -> dict:
    total_runs = len(DAY_TYPES) * N_SEEDS * len(SCHEDULER_MAP)
    if verbose:
        print(f"  Total simulation runs: {total_runs}")

    t0 = time.time()
    results = {}
    run_count = 0

    for day_type in DAY_TYPES:
        key = f"fleet{CONCURRENCY}_type{day_type}"
        results[key] = {
            "fleet_size":  CONCURRENCY,
            "day_type":    day_type,
            "n_jobs":      DAY_TYPE_PARAMS[day_type]["n_jobs"],
            "arrival_rate": DAY_TYPE_PARAMS[day_type]["arrival_rate"],
            "schedulers":  {},
        }

        for name, cls in SCHEDULER_MAP.items():
            seed_results = []
            for seed in range(N_SEEDS):
                seed_results.append(_run_one(cls, seed, day_type))
                run_count += 1

            wait_vals = [r["avg_wait_min"] for r in seed_results]
            miss_vals = [r["avg_miss_pct"] for r in seed_results]

            results[key]["schedulers"][name] = {
                "avg_wait_min":      _mean(wait_vals),
                "avg_miss_pct":      _mean(miss_vals),
                "total_cost":        _mean([r["total_cost"] for r in seed_results]),
                "utilization_pct":   _mean([r["utilization_pct"] for r in seed_results]),
                "avg_tardiness_min": _mean([r["avg_tardiness_min"] for r in seed_results]),
                "p95_tardiness_min": _mean([r["p95_tardiness_min"] for r in seed_results]),
            }

        if verbose:
            elapsed = time.time() - t0
            rate = run_count / elapsed if elapsed > 0 else 1
            remaining = (total_runs - run_count) / rate
            spt_wait = results[key]["schedulers"]["SPT"]["avg_wait_min"]
            edf_miss = results[key]["schedulers"]["EDF"]["avg_miss_pct"]
            rescue_miss = results[key]["schedulers"]["SPT+Rescue"]["avg_miss_pct"]
            print(
                f"  [{run_count}/{total_runs}] {day_type:>7}"
                f"  SPT={spt_wait:.1f}min  EDF_miss={edf_miss:.2f}%"
                f"  Rescue_miss={rescue_miss:.2f}%"
                f"  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)"
            )

    elapsed = time.time() - t0

    return {
        "metadata": {
            "concurrency":  CONCURRENCY,
            "day_types":    DAY_TYPES,
            "n_seeds":      N_SEEDS,
            "duration_hr":  DURATION / 3600,
            "total_runs":   total_runs,
            "elapsed_sec":  round(elapsed, 1),
        },
        "sensitivity": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CGRSP Phase 3 — Day-Type Sensitivity Analysis")
    _default_out = str(Path(__file__).parent / "Outputs/phase3/sensitivity.json")
    parser.add_argument("--output", default=_default_out)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 — DAY-TYPE SENSITIVITY (C_max=5, 30 seeds each)")
    print(f"  Day types    : {DAY_TYPES}")
    print(f"  Seeds/type   : {N_SEEDS}")
    print("=" * 60)

    output = run_sensitivity_analysis(verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\nAvg wait time (min) by day type:")
    print(f"  {'Day Type':<10}", end="")
    for name in SCHEDULER_MAP:
        print(f"  {name:>11}", end="")
    print()
    print("  " + "-" * (10 + 13 * len(SCHEDULER_MAP)))
    for day_type in DAY_TYPES:
        key = f"fleet{CONCURRENCY}_type{day_type}"
        print(f"  {day_type:<10}", end="")
        for name in SCHEDULER_MAP:
            val = output["sensitivity"][key]["schedulers"][name]["avg_wait_min"]
            print(f"  {val:>11.2f}", end="")
        print()


if __name__ == "__main__":
    main()
