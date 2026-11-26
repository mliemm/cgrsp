"""
Phase 3 — Deadline Tightness Sensitivity
==========================================
Varies the fraction of jobs with tight (1 h) vs loose (8 h) deadlines
to identify the load conditions under which EDF's deadline-safety advantage
over SPT becomes statistically visible.

Fractions swept: 10%, 30%, 50%, 70%, 90% tight deadlines.
Each: 730 jobs (surge day), C_max=5, arrival=0.020 jobs/sec, 30 seeds.

Output: Outputs/phase3/deadline.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys, os, json, math, time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers import (
    FIFOScheduler, SPTScheduler, EDFScheduler,
    LCFScheduler, BalancedScheduler, RandomScheduler, LaxityRescueScheduler,
)

SCHEDULER_MAP = {
    "FIFO": FIFOScheduler,
    "SPT": SPTScheduler,
    "EDF": EDFScheduler,
    "LCF": LCFScheduler,
    "Balanced": BalancedScheduler,
    "Random": RandomScheduler,
    "SPT+Rescue": LaxityRescueScheduler,
}

TIGHT_FRACTIONS = [0.10, 0.30, 0.50, 0.70, 0.90]
TIGHT_DEADLINE = 3600.0    # 1 hour
LOOSE_DEADLINE = 28800.0   # 8 hours
N_JOBS = 730          # surge-day job count (stress-load assumption; average surge-class billing day of about 750 jobs)
ARRIVAL_RATE = 0.020  # λ = 0.020 jobs/s  (surge day — shows deadline pressure)
CONCURRENCY_CONFIGS = [5]       # RunPod default account limit
DURATION = 86400.0
N_SEEDS = 30


def _paired_spt_edf(spt_miss, edf_miss):
    """Paired SPT vs EDF miss-rate test at one sweep point.

    All schedulers see identical generated jobs per seed, so the sweep is the
    same randomised block design as Phase 3 and the paired test applies here
    too. Reported so that a crossover in the point estimates is not mistaken
    for a demonstrated difference: at the crossover the two are expected to be
    statistically indistinguishable, which the sweep table alone cannot show.
    The sign convention matches phase3_statistical.py's "SPT vs EDF" entry, so
    t is negative when SPT misses fewer deadlines than EDF.
    """
    a, b = np.asarray(spt_miss, float), np.asarray(edf_miss, float)
    n = len(a)
    res = stats.ttest_rel(a, b)
    pooled_var = (float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0
    d = (float(a.mean()) - float(b.mean())) / math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    p = float(res.pvalue)
    return {
        "comparison": "SPT vs EDF",
        "metric": "deadline_miss_pct",
        "t": round(float(res.statistic), 4),
        "p": round(p, 6),
        "df": n - 1,
        "cohen_d": round(d, 4),
        "n_seeds": n,
        "significant_p05": p < 0.05,
        "raw_miss_spt": list(a),
        "raw_miss_edf": list(b),
    }


def _run_one(scheduler_cls, seed, tight_fraction, concurrency=5):
    deadline_dist = [
        (TIGHT_DEADLINE, tight_fraction),
        (LOOSE_DEADLINE, 1.0 - tight_fraction),
    ]
    job_config = JobGeneratorConfig(
        num_jobs=N_JOBS,
        arrival_rate=ARRIVAL_RATE,
        seed=seed,
        deadline_distribution=deadline_dist,
    )
    jobs = JobGenerator(job_config).generate()

    cloud = RunPodCloudFleetManager(
        max_concurrency=concurrency,
        supported_types=ResourceType.supported_gpus(),
        random_seed=seed,
    )
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sched = scheduler_cls(fleet_manager=cloud, seed=seed)
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
        "avg_wait_min": round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "deadline_miss_pct": round(m["job_counts"]["deadline_miss_rate_pct"], 4),
        "total_cost": round(m["cost_metrics"]["total_cost"], 6),
        "completed": m["job_counts"]["completed"],
    }


def run_deadline_sensitivity(verbose=True):
    total = len(CONCURRENCY_CONFIGS) * len(TIGHT_FRACTIONS) * len(SCHEDULER_MAP) * N_SEEDS
    if verbose:
        print(f"  Total runs: {total}")

    t0 = time.time()
    all_results = {}

    for concurrency in CONCURRENCY_CONFIGS:
        fleet_key = f"fleet{concurrency}"
        all_results[fleet_key] = {}

        for frac in TIGHT_FRACTIONS:
            key = f"tight{int(frac*100)}pct"
            all_results[fleet_key][key] = {"tight_fraction": frac, "schedulers": {}}

            for name, cls in SCHEDULER_MAP.items():
                seed_runs = [_run_one(cls, seed, frac, concurrency) for seed in range(N_SEEDS)]
                all_results[fleet_key][key]["schedulers"][name] = {
                    "avg_wait_min": round(float(np.mean([r["avg_wait_min"] for r in seed_runs])), 3),
                    "deadline_miss_pct": round(float(np.mean([r["deadline_miss_pct"] for r in seed_runs])), 3),
                    "total_cost": round(float(np.mean([r["total_cost"] for r in seed_runs])), 4),
                    "raw_miss": [r["deadline_miss_pct"] for r in seed_runs],
                }

            sched_out = all_results[fleet_key][key]["schedulers"]
            all_results[fleet_key][key]["spt_vs_edf_miss"] = _paired_spt_edf(
                sched_out["SPT"]["raw_miss"], sched_out["EDF"]["raw_miss"]
            )

            if verbose:
                elapsed = time.time() - t0
                s = {n: all_results[fleet_key][key]["schedulers"][n]["deadline_miss_pct"]
                     for n in SCHEDULER_MAP}
                tt = all_results[fleet_key][key]["spt_vs_edf_miss"]
                print(
                    f"  fleet={concurrency}  tight={int(frac*100):3d}%  "
                    f"miss: EDF={s['EDF']:.2f}% SPT={s['SPT']:.2f}% RND={s['Random']:.2f}%  "
                    f"SPT vs EDF t={tt['t']:+.3f} p={tt['p']:.4f}  "
                    f"({elapsed:.0f}s)"
                )

    elapsed = time.time() - t0
    return {
        "metadata": {
            "tight_fractions": TIGHT_FRACTIONS,
            "tight_deadline_sec": TIGHT_DEADLINE,
            "loose_deadline_sec": LOOSE_DEADLINE,
            "n_jobs": N_JOBS,
            "arrival_rate": ARRIVAL_RATE,
            "concurrency_configs": CONCURRENCY_CONFIGS,
            "n_seeds": N_SEEDS,
            "elapsed_sec": round(elapsed, 1),
        },
        "results": all_results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    _default_out = str(Path(__file__).parent / "Outputs/phase3/deadline.json")
    parser.add_argument("--output", default=_default_out)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 — DEADLINE TIGHTNESS SENSITIVITY")
    print(f"  Tight fractions : {[int(f*100) for f in TIGHT_FRACTIONS]}%")
    print(f"  Fleet configs   : {CONCURRENCY_CONFIGS} GPUs")
    print(f"  Arrival rate    : {ARRIVAL_RATE} jobs/sec")
    print(f"  Seeds / combo   : {N_SEEDS}")
    print("=" * 60)

    output = run_deadline_sensitivity(verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Quick summary table
    for fleet_key, fracs in output["results"].items():
        print(f"\nFleet: {fleet_key}")
        print(f"{'Tight%':>7}", end="")
        for name in SCHEDULER_MAP:
            print(f" {name[:5]:>7}", end="")
        print()
        print("-" * (7 + 8 * len(SCHEDULER_MAP)))
        for frac_key, cfg in fracs.items():
            print(f"{int(cfg['tight_fraction']*100):>6}%", end="")
            for name in SCHEDULER_MAP:
                miss = cfg["schedulers"][name]["deadline_miss_pct"]
                print(f" {miss:>7.2f}", end="")
            print()


if __name__ == "__main__":
    main()
