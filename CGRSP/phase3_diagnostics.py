"""
Phase 3: diagnostics behind three claims that previously rested on one-off scratch runs.

Hectic day (950 jobs, lambda=0.100, C_max=5, 20% tight 1h / 80% loose 8h), the same job
streams and fleet as the Phase 3 benchmark. Three parts:

A. Mean-based service. The log-normal service draw is replaced by its stratum mean
   (30 seeds, all ten policies) and compared with the stored Phase 3 results.
B. RH slot timeline. In an instrumented RH run (seeds 0 to 2) the share of decision
   events with at least one busy slot at which the soonest free time t_free equals the
   current instant, so the urgent tier is empty.
C. Per-seed misses and dispatch clock (seeds 0 and 42): for RH, CADR and SPT, the number
   of misses that are tight jobs, their mean tardiness, the dispatches that fall in the
   12:00 to 18:00 availability band (simulation time zero is 08:00), and when arrivals
   and dispatching end.

Output: Outputs/phase3/diagnostics.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.config import ResourceType
from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, ExecutionTimeModel, GPUType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.schedulers import RollingHorizonScheduler
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.simulator.core import EventType, Simulator
import phase3_statistical as p3

N_SEEDS = 30
DEADLINE_DIST = p3.DEADLINE_DIST
START_CLOCK_H = 8.0          # simulation time zero is 08:00
PEAK_BAND_S = ((12.0 - START_CLOCK_H) * 3600.0, (18.0 - START_CLOCK_H) * 3600.0)
TIGHT_WINDOW_S = 7200.0      # the scheduler's own definition of a tight job
STORED = Path(__file__).parent / "Outputs" / "phase3" / "statistical.json"
OUT = Path(__file__).parent / "Outputs" / "phase3" / "diagnostics.json"


def run_sim(scheduler_cls, seed):
    """Build and run one hectic-day simulation exactly as Phase 3 does; return the simulator."""
    jobs = JobGenerator(JobGeneratorConfig(
        num_jobs=p3.N_JOBS, arrival_rate=p3.ARRIVAL_RATE, seed=seed,
        deadline_distribution=DEADLINE_DIST)).generate()
    cloud = RunPodCloudFleetManager(
        max_concurrency=p3.CONCURRENCY, supported_types=ResourceType.supported_gpus(),
        random_seed=seed)
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sim.set_scheduler(scheduler_cls(fleet_manager=cloud, seed=seed))
    for job in jobs:
        sim.schedule_event(EventType.JOB_ARRIVAL, job.submit_time,
                           {"complexity": job.complexity, "deadline": job.deadline,
                            "estimated_duration": job.estimated_duration or 100.0})
    sim.run(duration=p3.DURATION)
    return sim, jobs


def summary(sim):
    m = sim.get_metrics_summary()
    return (m["timing_metrics"]["avg_waiting_time"] / 60.0,
            m["job_counts"]["deadline_miss_rate_pct"])


def part_a(stored):
    """Mean-based service against the stored log-normal results."""
    original = ExecutionTimeModel.sample_execution_time

    def mean_only(complexity, gpu_type, rng, sigma_ln=None):
        return EXECUTION_TIME_LOOKUP[(complexity, GPUType(gpu_type.value))]

    ExecutionTimeModel.sample_execution_time = staticmethod(mean_only)
    try:
        raw = {name: {"wait": [], "miss": []} for name in p3.SCHEDULER_MAP}
        for seed in range(N_SEEDS):
            for name, cls in p3.SCHEDULER_MAP.items():
                wait, miss = summary(run_sim(cls, seed)[0])
                raw[name]["wait"].append(wait)
                raw[name]["miss"].append(miss)
    finally:
        ExecutionTimeModel.sample_execution_time = original
    out = {}
    for name, arrays in raw.items():
        base_wait = np.mean(stored[name]["raw_wait_min"])
        base_miss = np.mean(stored[name]["raw_miss_pct"])
        wait, miss = float(np.mean(arrays["wait"])), float(np.mean(arrays["miss"]))
        out[name] = {
            "lognormal_wait_min": round(base_wait, 4), "mean_based_wait_min": round(wait, 4),
            "wait_change_pct": round(100.0 * (wait - base_wait) / base_wait, 3),
            "lognormal_miss_pct": round(base_miss, 4), "mean_based_miss_pct": round(miss, 4),
            "miss_change_pp": round(miss - base_miss, 4),
            "miss_change_rel_pct": round(100.0 * (miss - base_miss) / base_miss, 3),
            "raw_miss_pct": [round(x, 4) for x in arrays["miss"]],
        }
    rh, cadr = np.array(raw["RH"]["miss"]), np.array(raw["CADR"]["miss"])
    test = stats.ttest_rel(rh, cadr)
    out["RH_vs_CADR_miss"] = {"rh_minus_cadr_pp": round(float((rh - cadr).mean()), 4),
                              "t": round(float(test.statistic), 4), "p": round(float(test.pvalue), 4)}
    return out


def part_b(seeds=(0, 1, 2)):
    """Share of RH decision events (with a busy slot) at which t_free equals the current instant."""
    events = {"with_busy_slot": 0, "collapsed": 0}
    original = RollingHorizonScheduler._soonest_free

    def counting(self, current_time, slots):
        result = original(self, current_time, slots)
        if any(slot["busy"] for slot in slots):
            events["with_busy_slot"] += 1
            if result <= current_time + 1e-9:
                events["collapsed"] += 1
        return result

    RollingHorizonScheduler._soonest_free = counting
    per_seed = {}
    try:
        for seed in seeds:
            before = dict(events)
            run_sim(p3.SCHEDULER_MAP["RH"], seed)
            busy = events["with_busy_slot"] - before["with_busy_slot"]
            collapsed = events["collapsed"] - before["collapsed"]
            per_seed[str(seed)] = {"events_with_busy_slot": busy, "collapsed": collapsed,
                                   "collapsed_pct": round(100.0 * collapsed / busy, 2)}
    finally:
        RollingHorizonScheduler._soonest_free = original
    pooled = 100.0 * events["collapsed"] / events["with_busy_slot"]
    return {"per_seed": per_seed, "events_with_busy_slot": events["with_busy_slot"],
            "collapsed": events["collapsed"], "collapsed_pct": round(pooled, 2)}


def part_c(seeds=(0, 42), names=("RH", "CADR", "SPT")):
    out = {}
    for seed in seeds:
        for name in names:
            sim, jobs = run_sim(p3.SCHEDULER_MAP[name], seed)
            metrics = sim.metrics_collector
            missed = [j for j in metrics.get_completed_job_metrics() if not j.deadline_met]
            tight = [j for j in missed if (j.deadline - j.submit_time) < TIGHT_WINDOW_S]
            late_min = lambda js: float(np.mean([(j.completion_time - j.deadline) / 60.0 for j in js])) if js else 0.0
            dispatch_times = [t for _, _, t, _ in metrics.dispatch_log]
            in_band = sum(PEAK_BAND_S[0] <= t < PEAK_BAND_S[1] for t in dispatch_times)
            clock = lambda s: START_CLOCK_H + s / 3600.0
            out[f"{name}_seed{seed}"] = {
                "misses": len(missed), "tight_misses": len(tight),
                "loose_misses": len(missed) - len(tight),
                "tight_miss_mean_late_min": round(late_min(tight), 1),
                "loose_miss_mean_late_min": round(late_min([j for j in missed if j not in tight]), 1),
                "dispatches": len(dispatch_times), "dispatches_in_1200_1800_band": in_band,
                "arrivals_end_clock_h": round(clock(max(j.submit_time for j in jobs)), 2),
                "last_dispatch_clock_h": round(clock(max(dispatch_times)), 2),
            }
    return out


def main():
    t0 = time.time()
    stored = json.loads(STORED.read_text())["schedulers"]
    result = {"description": "Hectic-day diagnostics: A mean-based service, B RH t_free collapse, "
                             "C per-seed tight misses and dispatch clock. 950 jobs, C_max=5.",
              "n_seeds_part_a": N_SEEDS}
    result["part_b_rh_t_free"] = part_b()
    print("B", result["part_b_rh_t_free"], flush=True)
    result["part_c_seed_detail"] = part_c()
    for key, value in result["part_c_seed_detail"].items():
        print("C", key, value, flush=True)
    result["part_a_mean_based_service"] = part_a(stored)
    for key, value in result["part_a_mean_based_service"].items():
        print("A", key, {k: v for k, v in value.items() if k != "raw_miss_pct"}, flush=True)
    result["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT} ({result['runtime_s']}s)")


if __name__ == "__main__":
    main()
