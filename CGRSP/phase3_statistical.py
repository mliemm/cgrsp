"""
Phase 3 — Statistical Analysis
===============================
Runs 30 independent seeds per scheduler and computes 95% CIs and
paired t-tests with a Wilcoxon signed-rank backup. All schedulers run on the
same 30 seeds with identical generated jobs per seed, so observations are
paired (a randomised block design); a paired test is both correct and more
powerful than an unpaired Welch test on this blocked design.

Output: Outputs/phase3/statistical.json
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
    RollingHorizonScheduler, CostAwareDeadlineRiskScheduler, AdaptiveScheduler,
)

SCHEDULER_MAP = {
    "FIFO": FIFOScheduler,
    "SPT": SPTScheduler,
    "EDF": EDFScheduler,
    "LCF": LCFScheduler,
    "Balanced": BalancedScheduler,
    "Random": RandomScheduler,
    "SPT+Rescue": LaxityRescueScheduler,
    "RH": lambda fleet_manager, seed: RollingHorizonScheduler(
        fleet_manager=fleet_manager, seed=seed, arrival_rate=0.1),
    "CADR": lambda fleet_manager, seed: CostAwareDeadlineRiskScheduler(
        fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=True),
    "Adaptive": lambda fleet_manager, seed: AdaptiveScheduler(fleet_manager, seed=seed),
}
N_SEEDS = 30
# Hectic day: 950 jobs/day (saturating stress-load assumption, about ten times an average hectic-class billing day), burst λ=0.100 j/s, 20% tight 1h / 80% loose 8h deadlines
N_JOBS = 950          # hectic-day job count
ARRIVAL_RATE = 0.100  # λ = 0.100 jobs/s  (burst arrival)
DEADLINE_DIST = [(3600.0, 0.20), (28800.0, 0.80)]  # 20% tight 1h / 80% loose 8h
CONCURRENCY = 5       # concurrency budget; equals RunPod default per-account worker quota
DURATION = 86400.0    # full 24-hour window


# ─── statistics helpers ─────────────────────────────────────────────────────

def _ci95(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t_crit = float(stats.t.ppf(0.975, n - 1))  # exact two-sided 95% critical value, df = n-1
    margin = t_crit * std / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_lo": round(mean - margin, 4),
        "ci95_hi": round(mean + margin, 4),
        "n": n,
    }


def _paired_t(a, b):
    """Paired t-test (same seeds) with a Wilcoxon signed-rank backup.

    The two samples are matched on seed (identical generated jobs per seed), so
    the correct test is paired. Returns t, two-tailed p (exact Student-t,
    df = n-1), the Wilcoxon signed-rank p as a nonparametric robustness backup,
    and Cohen's d (pooled SD, identical to the unpaired definition so the
    effect-size magnitudes are directly comparable across analyses).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    diff = a - b
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))

    # Cohen's d (pooled SD, equal n): identical to the unpaired pooled-d value.
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled_var = (va + vb) / 2.0
    d = (float(a.mean()) - float(b.mean())) / math.sqrt(pooled_var) if pooled_var > 0 else 0.0

    if sd == 0.0:
        # Zero variance of the differences: either the schedulers are
        # literally identical on every seed (md == 0, no effect) or one beats
        # the other by exactly the same amount on every seed (a perfectly
        # consistent effect, t -> +/- inf). Unreachable in the canonical data.
        if md == 0.0:
            return {"t": 0.0, "p": 1.0, "df": n - 1, "cohen_d": round(d, 4),
                    "wilcoxon_p": 1.0, "significant_p05": False}
        return {"t": float("inf") if md > 0 else float("-inf"), "p": 0.0,
                "df": n - 1, "cohen_d": round(d, 4),
                "wilcoxon_p": 0.0, "significant_p05": True}

    t = md / (sd / math.sqrt(n))
    df = n - 1
    # Two-tailed p from the exact Student t-distribution. Matches an independent
    # recomputation with scipy.stats.ttest_rel(a, b).
    p = float(2.0 * stats.t.sf(abs(t), df))

    # Wilcoxon signed-rank: distribution-free confirmation. Falls back to NaN
    # only in the degenerate all-zero-difference case.
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")

    return {
        "t": round(t, 4),
        "p": round(p, 6),
        "df": df,
        "cohen_d": round(d, 4),
        "wilcoxon_p": round(w_p, 6),
        "significant_p05": p < 0.05,
    }


# ─── simulation runner ───────────────────────────────────────────────────────

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
    from cgrsp.data.execution_times import GPUType, SAMPLE_SIZES
    sparse_count = sum(
        SAMPLE_SIZES[(jobs[job_id].complexity, GPUType(cloud.get_node(node_id).resource_type.value))] < 5
        for job_id, node_id, _, _ in sim.metrics_collector.dispatch_log
    )
    return {
        "sparse_dispatch_pct": 100.0 * sparse_count / len(jobs),
        "avg_wait_min": round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "deadline_miss_pct": round(m["job_counts"]["deadline_miss_rate_pct"], 4),
        "total_cost": round(m["cost_metrics"]["total_cost"], 6),
        "utilization_pct": round(m["utilization"]["avg_pct"], 4),
        "completed": m["job_counts"]["completed"],
        "avg_tardiness_min": m.get("avg_tardiness_s", 0.0) / 60.0,
        "p95_tardiness_min": m.get("p95_tardiness_s", 0.0) / 60.0,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def run_statistical_analysis(n_seeds: int = N_SEEDS, verbose: bool = True) -> dict:
    seeds = list(range(n_seeds))
    raw: dict = {name: [] for name in SCHEDULER_MAP}
    raw_ablation: dict = {"CADR-order-only": []}

    t0 = time.time()
    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"  Seed {seed:2d}/{seeds[-1]} ...", end=" ", flush=True)
        t_seed = time.time()
        for name, cls in SCHEDULER_MAP.items():
            raw[name].append(_run_one(cls, seed))
        # CADR-order-only ablation: cost_aware=False disables cost-aware tie-breaking
        cadr_oo_cls = lambda fleet_manager, seed: CostAwareDeadlineRiskScheduler(  # noqa: E731
            fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=False)
        raw_ablation["CADR-order-only"].append(_run_one(cadr_oo_cls, seed))
        if verbose:
            print(f"{time.time() - t_seed:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nAll {n_seeds} seeds done in {elapsed:.1f}s.")

    # Aggregate per scheduler
    schedulers_out = {}
    for name, runs in raw.items():
        wait = [r["avg_wait_min"] for r in runs]
        miss = [r["deadline_miss_pct"] for r in runs]
        cost = [r["total_cost"] for r in runs]
        util = [r["utilization_pct"] for r in runs]
        tard_avg = [r["avg_tardiness_min"] for r in runs]
        tard_p95 = [r["p95_tardiness_min"] for r in runs]
        schedulers_out[name] = {
            "raw_wait_min": wait,
            "raw_miss_pct": miss,
            "raw_cost_usd": cost,
            "raw_util_pct": util,
            "raw_avg_tardiness_min": tard_avg,
            "raw_p95_tardiness_min": tard_p95,
            "wait_min": _ci95(wait),
            "miss_pct": _ci95(miss),
            "cost_usd": _ci95(cost),
            "util_pct": _ci95(util),
            "avg_tardiness_min": _ci95(tard_avg),
            "p95_tardiness_min": _ci95(tard_p95),
        }

    # Aggregate ablation variant (kept separate from main schedulers block)
    ablation_extra = {}
    for name, runs in raw_ablation.items():
        wait = [r["avg_wait_min"] for r in runs]
        miss = [r["deadline_miss_pct"] for r in runs]
        cost = [r["total_cost"] for r in runs]
        util = [r["utilization_pct"] for r in runs]
        tard_avg = [r["avg_tardiness_min"] for r in runs]
        tard_p95 = [r["p95_tardiness_min"] for r in runs]
        ablation_extra[name] = {
            "raw_wait_min": wait,
            "raw_miss_pct": miss,
            "raw_cost_usd": cost,
            "raw_util_pct": util,
            "raw_avg_tardiness_min": tard_avg,
            "raw_p95_tardiness_min": tard_p95,
            "wait_min": _ci95(wait),
            "miss_pct": _ci95(miss),
            "cost_usd": _ci95(cost),
            "util_pct": _ci95(util),
            "avg_tardiness_min": _ci95(tard_avg),
            "p95_tardiness_min": _ci95(tard_p95),
        }

    # Pairwise t-tests vs FIFO (reference baseline)
    tests = []
    reference = "FIFO"
    ref_wait = [r["avg_wait_min"] for r in raw[reference]]
    ref_miss = [r["deadline_miss_pct"] for r in raw[reference]]
    for name in SCHEDULER_MAP:
        if name == reference:
            continue
        cand_wait = [r["avg_wait_min"] for r in raw[name]]
        cand_miss = [r["deadline_miss_pct"] for r in raw[name]]
        tests.append({
            "comparison": f"{name} vs {reference}",
            "metric": "avg_wait_min",
            "test": _paired_t(cand_wait, ref_wait),
        })
        tests.append({
            "comparison": f"{name} vs {reference}",
            "metric": "deadline_miss_pct",
            "test": _paired_t(cand_miss, ref_miss),
        })

    # Also SPT vs EDF on miss rate (deadline-safety comparison)
    tests.append({
        "comparison": "SPT vs EDF",
        "metric": "deadline_miss_pct",
        "test": _paired_t(
            [r["deadline_miss_pct"] for r in raw["SPT"]],
            [r["deadline_miss_pct"] for r in raw["EDF"]],
        ),
    })

    # RH vs SPT+Rescue (ablation step comparison; FIFO comparison handled by loop above)
    rh_wait = [r["avg_wait_min"] for r in raw["RH"]]
    rh_miss = [r["deadline_miss_pct"] for r in raw["RH"]]
    rescue_wait = [r["avg_wait_min"] for r in raw["SPT+Rescue"]]
    rescue_miss = [r["deadline_miss_pct"] for r in raw["SPT+Rescue"]]
    tests.append({
        "comparison": "RH vs SPT+Rescue",
        "metric": "avg_wait_min",
        "test": _paired_t(rh_wait, rescue_wait),
    })
    tests.append({
        "comparison": "RH vs SPT+Rescue",
        "metric": "deadline_miss_pct",
        "test": _paired_t(rh_miss, rescue_miss),
    })

    # RH vs CADR on miss rate: the two proposed schedulers. Tests whether RH's
    # lower miss point estimate is a statistically distinguishable advantage or
    # whether the two are tied on miss (the recommendation then turns on
    # tardiness, not miss count).
    cadr_miss = [r["deadline_miss_pct"] for r in raw["CADR"]]
    tests.append({
        "comparison": "RH vs CADR",
        "metric": "deadline_miss_pct",
        "test": _paired_t(rh_miss, cadr_miss),
    })

    # RH vs SPT on wait: RH's headline is lowest miss AND lowest-tier wait, and
    # the second half of that claim rests on RH being indistinguishable from
    # SPT, the wait leader. Tested here rather than inferred from overlapping
    # marginal CIs, which is not a test on a randomised block design.
    spt_wait = [r["avg_wait_min"] for r in raw["SPT"]]
    tests.append({
        "comparison": "RH vs SPT",
        "metric": "avg_wait_min",
        "test": _paired_t(rh_wait, spt_wait),
    })

    # CADR vs SPT+Rescue on wait: the corresponding tie for the second proposed
    # scheduler, likewise previously asserted without a test.
    cadr_wait = [r["avg_wait_min"] for r in raw["CADR"]]
    tests.append({
        "comparison": "CADR vs SPT+Rescue",
        "metric": "avg_wait_min",
        "test": _paired_t(cadr_wait, rescue_wait),
    })

    output = {
        "metadata": {
            "n_seeds": n_seeds,
            "n_jobs": N_JOBS,
            "arrival_rate": ARRIVAL_RATE,
            "concurrency": CONCURRENCY,
            "duration_hr": DURATION / 3600,
            "elapsed_sec": round(elapsed, 1),
        },
        "schedulers": schedulers_out,
        "t_tests": tests,
        "ablation_extra": ablation_extra,
    }
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CGRSP Phase 3 — Statistical Analysis")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    _default_out = str(Path(__file__).parent / "Outputs/phase3/statistical.json")
    parser.add_argument("--output", default=_default_out)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 — STATISTICAL ANALYSIS")
    print(f"  Seeds per scheduler : {args.seeds}")
    print(f"  Jobs / run          : {N_JOBS}")
    print(f"  Arrival rate        : {ARRIVAL_RATE} jobs/s")
    print("=" * 60)

    results = run_statistical_analysis(n_seeds=args.seeds, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")

    # Quick summary table
    print("\n" + "=" * 60)
    print(f"{'Scheduler':<10} {'Avg Wait (min)':<20} {'Miss %':<18} {'Cost ($)':<18}")
    print("-" * 60)
    for name, s in results["schedulers"].items():
        w = s["wait_min"]
        m = s["miss_pct"]
        c = s["cost_usd"]
        print(
            f"{name:<10} "
            f"{w['mean']:.2f} ± {w['std']:.2f} (95%CI [{w['ci95_lo']:.2f},{w['ci95_hi']:.2f}])  "
        )
        print(
            f"{'':10} miss: {m['mean']:.2f}±{m['std']:.2f}%   cost: ${c['mean']:.4f}±{c['std']:.4f}"
        )

    print("\nT-tests (vs FIFO):")
    for t in results["t_tests"]:
        sig = "✓ sig" if t["test"]["significant_p05"] else "✗ n.s."
        print(
            f"  {t['comparison']:<20} {t['metric']:<20} "
            f"t={t['test']['t']:+.3f}  p={t['test']['p']:.4f}  "
            f"d={t['test']['cohen_d']:.3f}  {sig}"
        )


if __name__ == "__main__":
    main()
