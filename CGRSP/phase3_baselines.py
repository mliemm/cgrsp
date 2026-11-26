"""
Phase 3 - Classical and Shallow Learned Baselines
=================================================
Runs three classical baselines (HEFT reduced to LPT, Min-Min, Max-Min) and one
shallow tabular contextual bandit on the same 30-seed hectic workload as
phase3_statistical.py, then reports paired t-tests of each new baseline against
FIFO, RH and CADR.

The workload parameters are imported from phase3_statistical.py, not copied, so
the two files cannot drift: 950 jobs, lambda = 0.100 jobs/s, 20% tight 1 h and
80% loose 8 h deadlines, C_max = 5 slots, 24 h window, seeds 0-29. FIFO, RH and
CADR are re-run here with the same _run_one, so their per-seed rows are the same
observations that statistical.json reports.

Bandit protocol (frozen in cgrsp/schedulers/bandit_scheduler.py): training
seeds 500-514 with epsilon = 0.25, evaluation seeds 0-29 with epsilon = 0 and
the frozen table. "Bandit-random" is the same class with an empty table and
epsilon = 1.0, a uniform random arm per scheduling call, the control that
separates learning from the arm set's diversity.

The paired tests are reported raw and with a Holm step-down adjustment over the
whole family of 30 comparisons (5 baselines x 3 references x 2 metrics), since
30 tests at p < 0.05 would otherwise be expected to yield about 1.5 false
positives.

Output: Outputs/phase3/baselines.json

Wall time of the full experiment (30 evaluation seeds x 8 policies plus 15
bandit training episodes) on the shared reference machine: 227 s for the run
written to baselines.json, under CPython 3.14.0, PYTHONHASHSEED=0 via _repro.
Training is 11 s of that and the 30 x 8 evaluation is 216 s, so the evaluation
is the floor set by the protocol; the spread across runs is background load
from other work on the same machine.
"""

# Pins PYTHONHASHSEED=0 so two runs produce the same JSON.
import _repro  # noqa: F401
import sys, os, json, platform, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers.bandit_scheduler import (
    ARMS,
    BanditScheduler,
    EVAL_SEEDS,
    TRAIN_SEEDS,
    WAIT_TIEBREAK_PER_MIN,
    assert_disjoint_seeds,
    episode_reward,
    update_q_table,
)
from cgrsp.schedulers.cost_aware_deadline_risk import CostAwareDeadlineRiskScheduler
from cgrsp.schedulers.fifo_scheduler import FIFOScheduler
from cgrsp.schedulers.heft_lpt_scheduler import HEFTLPTScheduler
from cgrsp.schedulers.max_min_scheduler import MaxMinScheduler
from cgrsp.schedulers.min_min_scheduler import MinMinScheduler
from cgrsp.schedulers.rolling_horizon_scheduler import RollingHorizonScheduler
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.simulator.core import EventType, Simulator
from phase3_statistical import (
    ARRIVAL_RATE,
    CONCURRENCY,
    DEADLINE_DIST,
    DURATION,
    N_JOBS,
    _ci95,
    _paired_t,
    _run_one,
)

N_SEEDS = 30
TRAIN_EPSILON = 0.25

RAW_METRIC_KEYS = (
    "avg_wait_min",
    "deadline_miss_pct",
    "total_cost",
    "utilization_pct",
    "avg_tardiness_min",
    "p95_tardiness_min",
)

# New baselines and the references they are tested against.
NEW_BASELINES = ("HEFT", "MinMin", "MaxMin", "Bandit", "Bandit-random")
REFERENCE_SCHEDULERS = ("FIFO", "RH", "CADR")
TEST_METRICS = ("avg_wait_min", "deadline_miss_pct")


# Training

def _simulate(scheduler_factory, seed):
    """
    Run one episode and return (simulator, cloud fleet, jobs, scheduler).

    The setup mirrors phase3_statistical._run_one exactly; it is duplicated
    only because training needs the handle on the scheduler instance and the
    simulator, which _run_one does not return.
    """
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
    scheduler = scheduler_factory(cloud, seed)
    sim.set_scheduler(scheduler)

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
    return sim, cloud, jobs, scheduler


def train_bandit(
    train_seeds=TRAIN_SEEDS,
    epsilon: float = TRAIN_EPSILON,
    verbose: bool = True,
):
    """
    Train the shallow contextual bandit on seeds disjoint from evaluation.

    One episode per training seed. Within an episode the arm is chosen by
    epsilon-greedy from the current table; after the episode the reward is
    computed from the dispatch log and every visited (context, arm) pair is
    updated once by incremental mean. Returns the table, the visit counts and a
    per-seed log.
    """
    assert_disjoint_seeds(train_seeds, EVAL_SEEDS)

    q_table = {}
    counts = {}
    log = []

    for index, seed in enumerate(train_seeds):
        created = {}

        def factory(fleet_manager, seed, created=created):
            scheduler = BanditScheduler(
                fleet_manager,
                seed=seed,
                q_table=q_table,
                epsilon=epsilon,
            )
            created["scheduler"] = scheduler
            return scheduler

        t0 = time.time()
        sim, _cloud, _jobs, scheduler = _simulate(factory, seed)
        reward = episode_reward(
            sim.metrics_collector.dispatch_log,
            sim.metrics_collector.get_job_metrics,
        )
        pairs = list(dict.fromkeys(scheduler.decision_log))
        update_q_table(q_table, counts, pairs, reward)

        log.append({
            "seed": seed,
            "reward": reward,
            "decisions": len(scheduler.decision_log),
            "updated_pairs": len(pairs),
            "elapsed_sec": round(time.time() - t0, 2),
        })
        if verbose:
            print(
                f"  train {index + 1:2d}/{len(train_seeds)} seed {seed} "
                f"reward {reward:+.4f} pairs {len(pairs):4d} "
                f"({time.time() - t0:.1f}s)"
            )

    return q_table, counts, log


# Evaluation

def _evaluation_schedulers(q_table):
    """Policy map for the evaluation seeds; the bandit table is frozen here."""
    return {
        "HEFT": HEFTLPTScheduler,
        "MinMin": MinMinScheduler,
        "MaxMin": MaxMinScheduler,
        "Bandit": lambda fleet_manager, seed: BanditScheduler(
            fleet_manager=fleet_manager, seed=seed,
            q_table=q_table, epsilon=0.0),
        "Bandit-random": lambda fleet_manager, seed: BanditScheduler(
            fleet_manager=fleet_manager, seed=seed,
            q_table=None, epsilon=1.0),
        "FIFO": FIFOScheduler,
        "RH": lambda fleet_manager, seed: RollingHorizonScheduler(
            fleet_manager=fleet_manager, seed=seed, arrival_rate=0.1),
        "CADR": lambda fleet_manager, seed: CostAwareDeadlineRiskScheduler(
            fleet_manager=fleet_manager, seed=seed, cr_star=3.0, cost_aware=True),
    }


def _add_holm(tests):
    """
    Holm step-down adjustment over the whole family, in place.

    Sorted ascending by the reported p; the adjusted value is the running
    maximum of (m - rank) * p, capped at 1.
    """
    family_size = len(tests)
    order = sorted(range(family_size), key=lambda i: tests[i]["test"]["p"])
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (family_size - rank) * tests[index]["test"]["p"])
        running = max(running, value)
        tests[index]["test"]["p_holm"] = round(running, 6)
        tests[index]["test"]["holm_significant_p05"] = running < 0.05


def run_baselines(n_seeds: int = N_SEEDS, verbose: bool = True) -> dict:
    eval_seeds = list(range(n_seeds))
    assert_disjoint_seeds(TRAIN_SEEDS, eval_seeds)

    t_start = time.time()

    # Bandit training on the disjoint seed range, before any evaluation.
    t_train = time.time()
    if verbose:
        print(f"Training bandit on {len(TRAIN_SEEDS)} seeds "
              f"({TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}), epsilon={TRAIN_EPSILON} ...")
    q_table, _counts, training_log = train_bandit(verbose=verbose)
    train_elapsed = time.time() - t_train

    schedulers = _evaluation_schedulers(q_table)
    raw = {name: [] for name in schedulers}

    t_eval = time.time()
    for seed in eval_seeds:
        if verbose:
            print(f"  seed {seed:2d}/{eval_seeds[-1]} ...", end=" ", flush=True)
        t_seed = time.time()
        for name, factory in schedulers.items():
            metrics = _run_one(factory, seed)
            record = {"seed": seed}
            for key in RAW_METRIC_KEYS:
                record[key] = metrics[key]
            raw[name].append(record)
        if verbose:
            print(f"{time.time() - t_seed:.1f}s")
    eval_elapsed = time.time() - t_eval

    # Aggregate per policy, mirroring statistical.json's raw arrays and CIs.
    out = {}
    for name, runs in raw.items():
        wait = [r["avg_wait_min"] for r in runs]
        miss = [r["deadline_miss_pct"] for r in runs]
        cost = [r["total_cost"] for r in runs]
        util = [r["utilization_pct"] for r in runs]
        tard_avg = [r["avg_tardiness_min"] for r in runs]
        tard_p95 = [r["p95_tardiness_min"] for r in runs]
        out[name] = {
            "per_seed": runs,
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

    # Paired t-tests: each new baseline against each reference, per metric.
    tests = []
    for name in NEW_BASELINES:
        candidate = {r["seed"]: r for r in raw[name]}
        for reference in REFERENCE_SCHEDULERS:
            reference_rows = {r["seed"]: r for r in raw[reference]}
            for metric in TEST_METRICS:
                cand_values = [candidate[s][metric] for s in eval_seeds]
                ref_values = [reference_rows[s][metric] for s in eval_seeds]
                tests.append({
                    "comparison": f"{name} vs {reference}",
                    "metric": metric,
                    "test": _paired_t(cand_values, ref_values),
                })
    _add_holm(tests)

    elapsed = time.time() - t_start

    output = {
        "metadata": {
            "n_seeds": len(eval_seeds),
            "n_jobs": N_JOBS,
            "arrival_rate": ARRIVAL_RATE,
            "concurrency": CONCURRENCY,
            "duration_hr": DURATION / 3600,
            "interpreter": platform.python_version(),
            "interpreter_full": sys.version.split()[0],
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "elapsed_sec": round(elapsed, 1),
            "train_elapsed_sec": round(train_elapsed, 1),
            "eval_elapsed_sec": round(eval_elapsed, 1),
            "multiple_comparison": (
                f"Holm step-down over the {len(tests)} paired comparisons "
                "(5 new baselines x 3 references x 2 metrics); p_holm is "
                "reported next to the raw p in every test"
            ),
            "bandit": {
                "arms": list(ARMS),
                "train_seeds": {
                    "start": TRAIN_SEEDS[0],
                    "stop": TRAIN_SEEDS[-1],
                    "n": len(TRAIN_SEEDS),
                },
                "eval_seeds": {
                    "start": eval_seeds[0],
                    "stop": eval_seeds[-1],
                    "n": len(eval_seeds),
                },
                "epsilon_train": TRAIN_EPSILON,
                "epsilon_eval": 0.0,
                "epsilon_random_control": 1.0,
                "reward": (
                    "-deadline_miss_rate - "
                    f"{WAIT_TIEBREAK_PER_MIN} * mean_wait_min"
                ),
                "credit_assignment": (
                    "episode-level, assigned post hoc from the dispatch log; "
                    "one update per visited (context, arm) pair per episode"
                ),
                "q_table_entries": len(q_table),
            },
        },
        "schedulers": out,
        "bandit_q_table": q_table,
        "bandit_training": training_log,
        "t_tests": tests,
    }
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CGRSP Phase 3 - classical and shallow learned baselines")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    _default_out = str(Path(__file__).parent / "Outputs/phase3/baselines.json")
    parser.add_argument("--output", default=_default_out)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 - CLASSICAL AND SHALLOW LEARNED BASELINES")
    print(f"  Seeds per policy   : {args.seeds}")
    print(f"  Jobs / run         : {N_JOBS}")
    print(f"  Arrival rate       : {ARRIVAL_RATE} jobs/s")
    print(f"  Bandit training    : seeds {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}")
    print("=" * 60)

    results = run_baselines(n_seeds=args.seeds, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 78)
    print(f"{'Policy':<15} {'Wait (min)':>12} {'Miss (%)':>10} {'Cost ($)':>10} {'Util (%)':>9}")
    print("-" * 78)
    for name, s in results["schedulers"].items():
        print(
            f"{name:<15} "
            f"{s['wait_min']['mean']:>12.2f} "
            f"{s['miss_pct']['mean']:>10.2f} "
            f"{s['cost_usd']['mean']:>10.4f} "
            f"{s['util_pct']['mean']:>9.2f}"
        )

    print("\nPaired t-tests of the new baselines (Holm over the 30 comparisons):")
    for t in results["t_tests"]:
        sig = "sig" if t["test"]["holm_significant_p05"] else "n.s."
        print(
            f"  {t['comparison']:<24} {t['metric']:<18} "
            f"t={t['test']['t']:+.3f}  p={t['test']['p']:.4f}  "
            f"p_holm={t['test']['p_holm']:.4f}  d={t['test']['cohen_d']:+.3f}  {sig}"
        )


if __name__ == "__main__":
    main()
