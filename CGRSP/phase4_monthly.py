"""
CGRSP Phase 4 — Monthly Simulation
=====================================
Simulates a 30-day month with a billing-derived day-type calendar and
heterogeneous daily workloads, drawing on two empirical sources:

  Source A (rendering job dataset, 6,234 jobs, 2025-07 to 2026-06): the
    per-stratum execution-time model only. The per-day job counts and arrival
    rates in DAY_PARAMS are modelling assumptions, not statistics of this
    record (its busiest day holds 415 jobs, 18 Sep 2025, Taiwan time):
    hectic  : λ=0.100 j/s,  ~950 jobs (assumed saturating stress load)
    surge   : λ=0.020 j/s,  ~730 jobs (assumed stress load)
    normal  : λ=0.002 j/s,  ~100 jobs (of the order observed)
    quiet   : λ=0.0008 j/s, ~6 jobs   (of the order observed)
    Day-type frequencies come from Source B, below.

  Source B — May 2026 RunPod billing explorer (export Apr 22 to May 22; 30 complete days Apr 22 to May 21 used):
    surge   (>$5 compute/day)    : 7% of days (May 7: $7.15, May 20: $17.57)
    hectic  ($0.95–5/day)        : 20% of days (avg $1.42)
    normal  ($0.05–0.95/day)     : 53% of days (avg $0.45)
    quiet   (<$0.05/day)         : 20% of days ($0 on all Sundays)

  DOW probabilities re-calibrated from billing data (n=30 complete days):
    Mon: S=0%  H=25% N=75% Q=0%
    Tue: S=0%  H=25% N=50% Q=25%
    Wed: S=20% H=20% N=60% Q=0%
    Thu: S=20% H=40% N=40% Q=0%
    Fri: S=0%  H=0%  N=75% Q=25%
    Sat: S=0%  H=25% N=75% Q=0%    (studio works Saturdays)
    Sun: S=0%  H=0%  N=0%  Q=100%  (Sunday always quiet)

  All day types use heterogeneous deadlines: 20% tight (1 h) / 80% loose (8 h).

Alongside the ten shipped policies the run reports one ablation arm, RH with
capacity reservation disabled, so the monthly cost of anticipation is measured
rather than assumed (see the RH_NORES_NAME block below). It is recorded under
the JSON's "reservation_ablation" key and is not part of the headline table.

Output: CGRSP/Outputs/phase4/monthly_results.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys
import os
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Dict

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

# ── Constants ─────────────────────────────────────────────────────────────────

CONCURRENCY      = 5        # RunPod default account limit
DURATION         = 86400.0  # seconds per day
N_DAYS           = 30
START_DOW        = 0        # 0 = Monday
N_MONTHLY_SEEDS  = 10       # independent monthly replications

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

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Reservation ablation arm (reported alongside, never a default change) ─────
#
# RH's anticipation is *not* dormant at monthly scale. Its load gate suppresses
# reservation only at rho >= 0.95, i.e. lambda >= 0.0627 at C_max=5, and of the
# four day types only hectic (lambda=0.100, rho=1.52) clears it; surge, normal
# and quiet sit at rho 0.303, 0.0303 and 0.0121 respectively, so reservation is
# live on roughly five sixths of the calendar. This arm re-runs RH with
# reservation_aggressiveness=0 over the identical schedules and seeds, giving a
# paired measurement of the monthly anticipation penalty instead of an assertion
# that there is none. It is an extra reported arm only: it is deliberately kept
# out of SCHEDULER_MAP so the shipped RH configuration, and every table, figure
# and macro derived from it, is untouched.

RH_NORES_NAME = "RH-NoReservation"


def _rh_no_reservation(fleet_manager, seed, arrival_rate):
    """RH with anticipation disabled (reservation_aggressiveness=0)."""
    return RollingHorizonScheduler(
        fleet_manager=fleet_manager, seed=seed, arrival_rate=arrival_rate,
        reservation_aggressiveness=0.0)

# ── Day-type calibration (Source B: May 2026 billing, 30 complete days) ───────
#
# Cost ratio surge/normal ≈ 27×, hectic/normal ≈ 3.2×.
# Surge avg billing $12.36 vs normal avg $0.45; real_cost/sim_cost ≈ 2.53 (the A5000
# Serverless Flex / Pods securePrice rate ratio). Job counts are stress-load assumptions,
# not counts from the job record. Surge (730) matches the average spend of the two
# surge-class billing days ($12.36, about 750 jobs at Serverless Flex rates). Hectic (950)
# is a saturating burst of the order of the single busiest billing day ($17.57 on May 20,
# about 1,070 jobs), roughly ten times an average hectic-class day ($1.42, about 90 jobs).
# The busiest day in the 6,234-job record holds 415 jobs (18 Sep 2025, Taiwan time).
#
# All day types use uniform 20% tight / 80% loose deadline mix.

DAY_PARAMS: Dict[str, Dict] = {
    "surge":  {
        "jobs_mean": 730, "jobs_std": 100,
        "lambda": 0.020,
        "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)],  # 20% tight 1h / 80% loose 8h
    },
    "hectic": {
        "jobs_mean": 950, "jobs_std": 80,
        "lambda": 0.100,
        "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)],  # 20% tight 1h / 80% loose 8h
    },
    "normal": {
        "jobs_mean": 100, "jobs_std": 25,
        "lambda": 0.002,
        "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)],  # 20% tight 1h / 80% loose 8h
    },
    "quiet":  {
        "jobs_mean":   6, "jobs_std":  3,
        "lambda": 0.0008,
        "deadline_dist": [(3600.0, 0.20), (28800.0, 0.80)],  # 20% tight 1h / 80% loose 8h
    },
}

# (surge, hectic, normal, quiet) — calibrated from 30 billing days
DOW_TYPE_PROBS: Dict[int, tuple] = {
    0: (0.00, 0.25, 0.75, 0.00),  # Mon
    1: (0.00, 0.25, 0.50, 0.25),  # Tue
    2: (0.20, 0.20, 0.60, 0.00),  # Wed
    3: (0.20, 0.40, 0.40, 0.00),  # Thu
    4: (0.00, 0.00, 0.75, 0.25),  # Fri
    5: (0.00, 0.25, 0.75, 0.00),  # Sat (studio works)
    6: (0.00, 0.00, 0.00, 1.00),  # Sun (always quiet)
}
DAY_TYPES = ["surge", "hectic", "normal", "quiet"]

OUT_DIR = Path(__file__).parent / "Outputs" / "phase4"


# ── Day schedule dataclass ─────────────────────────────────────────────────────

@dataclass
class DayConfig:
    day_index: int
    dow: int
    day_type: str
    n_jobs: int
    arrival_rate: float
    deadline_dist: list


# ── Schedule generation ────────────────────────────────────────────────────────

def generate_monthly_schedule(seed: int, n_days: int = N_DAYS,
                               start_dow: int = START_DOW) -> List[DayConfig]:
    rng = np.random.default_rng(seed)
    schedule: List[DayConfig] = []
    for i in range(n_days):
        dow = (start_dow + i) % 7
        probs = DOW_TYPE_PROBS[dow]
        day_type = rng.choice(DAY_TYPES, p=probs)
        p = DAY_PARAMS[day_type]
        n_jobs = max(1, int(rng.normal(p["jobs_mean"], p["jobs_std"])))
        schedule.append(DayConfig(
            day_index=i,
            dow=dow,
            day_type=day_type,
            n_jobs=n_jobs,
            arrival_rate=p["lambda"],
            deadline_dist=p["deadline_dist"],
        ))
    return schedule


# Calendar simulation and submission-cohort aggregation

class _CalendarScheduler:
    """Keep policy state across days and update RH's distributional load input."""

    def __init__(self, scheduler, schedule):
        self.scheduler = scheduler
        self.days = schedule

    def schedule(self, jobs, current_time):
        if hasattr(self.scheduler, "arrival_rate"):
            index = int(current_time // DURATION)
            self.scheduler.arrival_rate = (
                self.days[index].arrival_rate if index < len(self.days) else 0.0
            )
        return self.scheduler.schedule(jobs, current_time)


def _day_jobs(day, seed):
    """Generate the requested trace, admitting arrivals within this calendar day."""
    config = JobGeneratorConfig(num_jobs=day.n_jobs, arrival_rate=day.arrival_rate,
                                seed=seed, deadline_distribution=day.deadline_dist)
    return [job for job in JobGenerator(config).generate() if job.submit_time < DURATION]


def _cohort_metrics(jobs):
    """Compute count-based means and a pooled nearest-rank tardiness percentile."""
    count = len(jobs)
    assert all(job.completion_time is not None for job in jobs), "Unfinished cohort"
    tardiness = [job.get_tardiness() / 60.0 for job in jobs]
    return {
        "n_jobs": count,
        "completed": count,
        "deadline_missed": sum(not job.deadline_met for job in jobs),
        "avg_wait_min": sum(job.waiting_time for job in jobs) / max(count, 1) / 60.0,
        "deadline_miss_pct": 100 * sum(not job.deadline_met for job in jobs) / max(count, 1),
        "total_cost": sum(job.execution_cost for job in jobs),
        "avg_tardiness_min": sum(tardiness) / max(count, 1),
        "p95_tardiness_min": (sorted(tardiness)[math.ceil(.95 * count) - 1] if count else 0.0),
    }


def run_month(scheduler_name: str, scheduler_cls,
              schedule: List[DayConfig], base_seed: int) -> Dict:
    """Run one continuous calendar, then drain accepted jobs beyond its end.

    A day's requested job count caps its Poisson trace. Arrivals at or after the
    next day boundary are outside that day's trace and are counted as excluded.
    Queued and running jobs persist across boundaries; all metrics are grouped
    by submission day and include completion after the final calendar boundary.
    """
    if not schedule or [day.day_index for day in schedule] != list(range(len(schedule))):
        raise ValueError("Require a nonempty schedule with consecutive day indices from zero")
    seed = base_seed * 1000
    cloud = RunPodCloudFleetManager(max_concurrency=CONCURRENCY,
                                   supported_types=ResourceType.supported_gpus(),
                                   random_seed=seed)
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    policy = scheduler_cls(cloud, seed, schedule[0].arrival_rate)
    sim.set_scheduler(_CalendarScheduler(policy, schedule))
    accepted_counts = []
    for day in schedule:
        jobs = _day_jobs(day, seed + day.day_index)
        accepted_counts.append(len(jobs))
        offset = day.day_index * DURATION
        for job in jobs:
            sim.schedule_event(EventType.JOB_ARRIVAL, offset + job.submit_time,
                               {"complexity": job.complexity,
                                "deadline": offset + job.deadline,
                                "estimated_duration": job.estimated_duration or 100.0})
    sim.run(duration=len(schedule) * DURATION, drain=True)
    collector = sim.metrics_collector
    all_jobs = collector.get_all_job_metrics()
    expected = sum(accepted_counts)
    assert len(all_jobs) == collector.system_metrics.total_jobs_completed == expected
    cohorts = [[] for _ in schedule]
    for job in all_jobs:
        cohorts[int(job.submit_time // DURATION)].append(job)
    samples = collector.get_utilization_time_series()
    daily = []
    for day, jobs in zip(schedule, cohorts):
        row = _cohort_metrics(jobs)
        row.update({"day_index": day.day_index, "dow": DOW_NAMES[day.dow],
                    "day_type": day.day_type, "arrival_rate": day.arrival_rate,
                    "requested_jobs": day.n_jobs,
                    "excluded_arrivals": day.n_jobs - len(jobs)})
        values = [value for timestamp, value in samples
                  if day.day_index * DURATION <= timestamp < (day.day_index + 1) * DURATION]
        row["utilization_pct"] = 100 * float(np.mean(values)) if values else 0.0
        daily.append(row)
    breakdown = {}
    for day_type in DAY_TYPES:
        selected = [job for day, jobs in zip(schedule, cohorts)
                    if day.day_type == day_type for job in jobs]
        n_days = sum(day.day_type == day_type for day in schedule)
        metrics = _cohort_metrics(selected)
        breakdown[day_type] = {
            "n_days": n_days, "total_jobs": metrics["n_jobs"],
            "avg_wait_min": metrics["avg_wait_min"] if n_days else None,
            "deadline_miss_pct": metrics["deadline_miss_pct"] if n_days else None,
            "total_cost": metrics["total_cost"] if n_days else 0,
        }
    monthly = _cohort_metrics(all_jobs)
    monthly.update({"scheduler": scheduler_name, "total_jobs": monthly.pop("n_jobs"),
                    "requested_jobs": sum(day.n_jobs for day in schedule),
                    "excluded_arrivals": sum(row["excluded_arrivals"] for row in daily),
                    "completion_time_s": max((job.completion_time for job in all_jobs), default=0.0),
                    "avg_util_pct": float(np.mean([row["utilization_pct"] for row in daily])),
                    "breakdown_by_day_type": breakdown, "daily_results": daily})
    return monthly


def run_day(scheduler_name: str, scheduler_cls, day: DayConfig, seed: int) -> Dict:
    """Run a one-day calendar with the same cutoff and completion-drain policy."""
    return run_month(scheduler_name, scheduler_cls, [replace(day, day_index=0)], seed)["daily_results"][0]

# ── Cross-replication statistics ───────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict:
    arr = np.array(vals, dtype=float)
    n = len(arr)
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if n > 1 else 0.0
    # Exact t_{0.975, n-1}. The previous 3-key lookup fell back to the normal
    # 1.96 for any other n, and the surge-day breakdown runs at n=9 once a
    # replication drawing zero surge days is filtered out, so those intervals
    # were about 15% too narrow.
    t_crit = float(stats.t.ppf(0.975, n - 1))
    margin = t_crit * s / math.sqrt(n)
    return {"mean": round(m, 4), "std": round(s, 4),
            "ci95_lo": round(m - margin, 4), "ci95_hi": round(m + margin, 4)}


def _paired_delta(default_vals: List[float], ablated_vals: List[float]) -> Dict:
    """Paired test of the shipped RH arm against the reservation-disabled arm.

    Both arms see the same schedules and the same calendar seeds and daily arrival streams, so the
    replications pair one-to-one and the paired test is the correct one. The
    delta is signed default minus ablated: positive on wait means anticipation
    costs wait, negative on miss means anticipation buys miss.
    """
    n = len(default_vals)
    diffs = np.array(default_vals, dtype=float) - np.array(ablated_vals, dtype=float)
    out = {
        "n": n,
        "mean_default": round(float(np.mean(default_vals)), 4),
        "mean_ablated": round(float(np.mean(ablated_vals)), 4),
        "mean_delta": round(float(diffs.mean()), 4),
    }
    if n > 1:
        t_stat, p_val = stats.ttest_rel(default_vals, ablated_vals)
        out["t"] = round(float(t_stat), 4)
        out["p_value"] = float(p_val)
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CGRSP Phase 4 — Monthly Simulation (with surge days)")
    print(f"  {N_DAYS} days × {N_MONTHLY_SEEDS} replications × {len(SCHEDULER_MAP)} schedulers")
    print(f"  Day types: surge/hectic/normal/quiet  C_max={CONCURRENCY}")
    print("=" * 65)

    results: Dict[str, List[Dict]] = {n: [] for n in SCHEDULER_MAP}
    ablation: List[Dict] = []          # RH with reservation disabled
    schedules: List[List[DayConfig]] = []

    t0 = time.time()
    for rep in range(N_MONTHLY_SEEDS):
        schedule = generate_monthly_schedule(seed=rep)
        schedules.append(schedule)

        # Print calendar: S=surge H=hectic N=normal Q=quiet
        sym = {"surge": "S", "hectic": "H", "normal": "N", "quiet": "Q"}
        row = [f"{sym[d.day_type]}({d.n_jobs:3d})" for d in schedule]
        n_surge = sum(1 for d in schedule if d.day_type == "surge")
        print(f"\nRep {rep}: {n_surge} surge day(s)")
        for i in range(0, N_DAYS, 7):
            week = schedule[i:i+7]
            print("  " + "  ".join(
                f"{DOW_NAMES[d.dow][:2]}:{sym[d.day_type]}{d.n_jobs:3d}"
                for d in week
            ))

        for name, cls in SCHEDULER_MAP.items():
            print(f"  [{rep+1}/{N_MONTHLY_SEEDS}] {name} ...", end=" ", flush=True)
            t_s = time.time()
            r = run_month(name, cls, schedule, base_seed=rep + 1)
            results[name].append(r)
            elapsed = time.time() - t_s
            # Flag surge breakdown if available
            s_bd = r["breakdown_by_day_type"].get("surge", {})
            surge_str = ""
            if s_bd.get("n_days", 0) > 0:
                surge_str = (f"  [surge: wait={s_bd['avg_wait_min']:.0f}min "
                             f"miss={s_bd['deadline_miss_pct']:.1f}%]")
            print(f"wait={r['avg_wait_min']:.1f}min  "
                  f"miss={r['deadline_miss_pct']:.2f}%  "
                  f"({elapsed:.1f}s){surge_str}")

        # Ablation arm: identical schedule and seeds, anticipation off. Run after
        # the map using the same calendar and seed as the shipped RH arm.
        print(f"  [{rep+1}/{N_MONTHLY_SEEDS}] {RH_NORES_NAME} ...", end=" ", flush=True)
        t_s = time.time()
        r_abl = run_month(RH_NORES_NAME, _rh_no_reservation, schedule, base_seed=rep + 1)
        ablation.append(r_abl)
        print(f"wait={r_abl['avg_wait_min']:.1f}min  "
              f"miss={r_abl['deadline_miss_pct']:.2f}%  "
              f"({time.time() - t_s:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # ── Summary ────────────────────────────────────────────────────────────────
    summary: Dict[str, Dict] = {}
    for name in SCHEDULER_MAP:
        reps = results[name]
        summary[name] = {
            "avg_wait_min":      _stats([r["avg_wait_min"]      for r in reps]),
            "deadline_miss_pct": _stats([r["deadline_miss_pct"] for r in reps]),
            "total_cost":        _stats([r["total_cost"]        for r in reps]),
            "avg_util_pct":      _stats([r["avg_util_pct"]      for r in reps]),
            "avg_tardiness_min": _stats([r["avg_tardiness_min"] for r in reps]),
            "p95_tardiness_min": _stats([r["p95_tardiness_min"] for r in reps]),
        }
        # Per-day-type CI across replications
        for dt in DAY_TYPES:
            waits  = [r["breakdown_by_day_type"][dt]["avg_wait_min"]
                      for r in reps if r["breakdown_by_day_type"][dt]["n_days"] > 0]
            misses = [r["breakdown_by_day_type"][dt]["deadline_miss_pct"]
                      for r in reps if r["breakdown_by_day_type"][dt]["n_days"] > 0]
            if waits:
                summary[name][f"{dt}_wait_min"] = _stats(waits)
                summary[name][f"{dt}_miss_pct"] = _stats(misses)

    # ── Reservation ablation summary ───────────────────────────────────────────
    rh_reps = results["RH"]
    ablation_block = {
        "description": (
            "RH re-run with reservation_aggressiveness=0 (anticipation off) on the "
            "identical schedules and calendar seeds and daily arrival streams. Reported arm only; the shipped "
            "RH configuration is unchanged. The load gate suppresses reservation "
            "only at rho >= 0.95 (lambda >= 0.0627 at C_max=5), so anticipation is "
            "inactive on hectic days alone and live on surge, normal and quiet days."
        ),
        "gate": {
            "rho_gate": 0.95,
            "lambda_gate": 0.0627,
            "rho_by_day_type": {"surge": 0.303, "hectic": 1.52,
                                "normal": 0.0303, "quiet": 0.0121},
            "reservation_active_day_types": ["surge", "normal", "quiet"],
        },
        "summary": {
            "avg_wait_min":      _stats([r["avg_wait_min"]      for r in ablation]),
            "deadline_miss_pct": _stats([r["deadline_miss_pct"] for r in ablation]),
            "total_cost":        _stats([r["total_cost"]        for r in ablation]),
            "avg_util_pct":      _stats([r["avg_util_pct"]      for r in ablation]),
            "avg_tardiness_min": _stats([r["avg_tardiness_min"] for r in ablation]),
            "p95_tardiness_min": _stats([r["p95_tardiness_min"] for r in ablation]),
        },
        "paired_vs_default": {
            metric: _paired_delta([r[metric] for r in rh_reps],
                                  [r[metric] for r in ablation])
            for metric in ("avg_wait_min", "deadline_miss_pct",
                           "total_cost", "avg_tardiness_min")
        },
        "replications": ablation,
    }

    # Schedule characterisation across all replications
    all_types = [d.day_type for s in schedules for d in s]
    schedule_stats = {
        dt: {"count": all_types.count(dt),
             "pct":   round(100 * all_types.count(dt) / len(all_types), 1)}
        for dt in DAY_TYPES
    }

    out = {
        "metadata": {
            "experiment": "phase4_monthly",
            "n_days": N_DAYS, "n_monthly_seeds": N_MONTHLY_SEEDS,
            "concurrency": CONCURRENCY,
            "calendar_policy": "Daily capped Poisson traces cut at 86400 s; persistent fleet and queue; drain after month; submission-cohort metrics.",
            "percentile_method": "Nearest rank on pooled completed-job tardiness.",
            "calibration_sources": [
                "data/rendering_performance_data.csv (6,234 jobs, 2025-07 to 2026-06)",
                "RunPod billing explorer Apr 22–May 21 2026 (30 complete days)",
            ],
            "surge_calibration": {
                "trigger": "billing >$5 compute/day (May 7: $7.15, May 20: $17.57)",
                "frequency": "6.7% of days (2/30)",
                "dow_surge_prob": {"Wed": 0.20, "Thu": 0.20, "others": 0.00},
                "jobs_mean": DAY_PARAMS["surge"]["jobs_mean"],
                "lambda": DAY_PARAMS["surge"]["lambda"],
                "deadline_dist": "20% tight 3600s / 80% loose 28800s",
            },
        },
        "schedule_stats": schedule_stats,
        "summary": summary,
        "replications": {name: results[name] for name in SCHEDULER_MAP},
        "reservation_ablation": ablation_block,
    }

    out_path = OUT_DIR / "monthly_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"\nSaved → {out_path}")

    # ── Comparison table ───────────────────────────────────────────────────────
    print("\n── Monthly Summary (30-day, 10 reps, with surge days) ─────────────")
    print(f"{'Sched':<10} {'Wait (min)':<22} {'Miss %':<22} {'Cost ($)'}")
    print("-" * 72)
    for name in SCHEDULER_MAP:
        s = summary[name]
        w = s["avg_wait_min"]; m = s["deadline_miss_pct"]; c = s["total_cost"]
        print(f"{name:<10} "
              f"{w['mean']:6.2f} [{w['ci95_lo']:.2f},{w['ci95_hi']:.2f}]  "
              f"{m['mean']:5.2f} [{m['ci95_lo']:.2f},{m['ci95_hi']:.2f}]  "
              f"{c['mean']:7.2f}")

    print("\n── RH reservation ablation (anticipation on vs off, paired) ───────")
    for metric, label in (("avg_wait_min", "Wait (min)"),
                          ("deadline_miss_pct", "Miss %"),
                          ("avg_tardiness_min", "Tardiness (min)")):
        d = ablation_block["paired_vs_default"][metric]
        tail = (f"  t={d['t']:+.3f}  p={d['p_value']:.4g}"
                if "t" in d else "  (n<2, no test)")
        print(f"  {label:<16} on={d['mean_default']:8.3f}  "
              f"off={d['mean_ablated']:8.3f}  delta={d['mean_delta']:+8.3f}{tail}")

    print("\n── Per-day-type breakdown (mean across replications) ─────────────")
    for dt in DAY_TYPES:
        print(f"\n  {dt.upper()}:")
        for name in SCHEDULER_MAP:
            wk = f"{dt}_wait_min"; mk = f"{dt}_miss_pct"
            if wk in summary[name]:
                w = summary[name][wk]["mean"]
                m = summary[name][mk]["mean"]
                print(f"    {name:<10} wait={w:6.2f} min   miss={m:5.2f}%")


if __name__ == "__main__":
    main()
