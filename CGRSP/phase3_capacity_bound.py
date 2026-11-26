"""
Phase 3 — Capacity Lower Bound and Clairvoyant Relaxation
==========================================================
Brackets the offline optimum miss count for the hectic-day instances.

Lower bracket (certified): a window-capacity bound on the expected-service
instance. For any window [s, t], the jobs submitted at or after s with
deadlines at or before t must fit their fastest-GPU expected work into
C_max * (t - s) machine-seconds; if the demand exceeds capacity, at least
ceil((demand - capacity) / max_job_work) of those jobs miss under ANY
policy. The relaxation grants every job its fastest GPU, zero provisioning
delay, and free preemption, so the bound is valid for every policy on the
deterministic expected-service instance (realised stochastic runs can differ
in both directions; the JSON description states the instance explicitly).

Upper bracket (constructive): clairvoyant non-preemptive EDF on the same
relaxed instance (C_max identical fastest-GPU machines, deterministic mean
service times, zero provisioning). Its realised miss count upper-bounds the
relaxed offline optimum.

Hectic day (950 jobs, λ=0.100), C_max=5, 30 seeds (same job streams as the
Phase 3 statistical benchmark).

Output: Outputs/phase3/capacity_bound.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys, os, json, heapq, math
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, GPUType

CONCURRENCY = 5
N_SEEDS     = 30
N_JOBS      = 950
ARRIVAL     = 0.100
DEADLINES   = [(3600.0, 0.20), (28800.0, 0.80)]

MIN_EXEC = {
    k: min(EXECUTION_TIME_LOOKUP[(k, g)] for g in GPUType)
    for k in (1, 2, 3)
}


def gen_jobs(seed):
    cfg = JobGeneratorConfig(num_jobs=N_JOBS, arrival_rate=ARRIVAL, seed=seed,
                             deadline_distribution=DEADLINES)
    return [(j.submit_time, j.deadline, MIN_EXEC[j.complexity])
            for j in JobGenerator(cfg).generate()]


def capacity_bound(jobs):
    """Max over windows [s, t] of the certified minimum number of misses."""
    submits = sorted({j[0] for j in jobs})
    best = 0
    for s in submits:
        window = sorted((d, w) for (r, d, w) in jobs if r >= s)
        demand = 0.0
        max_w = 0.0
        for d, w in window:
            demand += w
            max_w = max(max_w, w)
            slack = demand - CONCURRENCY * (d - s)
            if slack > 1e-9:
                # subtract a float-noise epsilon so accumulated summation error
                # can never push the ratio past an exact integer and produce an
                # invalid (over-certified) bound
                best = max(best, math.ceil(slack / max_w - 1e-9))
    return best


def relaxed_edf_misses(jobs):
    """Non-preemptive EDF list schedule on C_max identical fastest machines,
    zero provisioning, deterministic mean service times, full hindsight of
    release times (jobs still cannot start before submission).

    The loop dispatches every ready job that can start at the current instant
    before advancing time, so simultaneous releases are served on all idle
    machines at once."""
    pending = sorted(jobs, key=lambda j: (j[0], j[1]))  # by release
    machines = [0.0] * CONCURRENCY  # next-free times
    ready = []  # (deadline, release, work)
    misses = 0
    i = 0
    t = 0.0
    EPS = 1e-9
    while i < len(pending) or ready:
        while i < len(pending) and pending[i][0] <= t + EPS:
            r, d, w = pending[i]
            heapq.heappush(ready, (d, r, w))
            i += 1
        if ready:
            free = min(range(CONCURRENCY), key=machines.__getitem__)
            if machines[free] <= t + EPS:
                d, r, w = heapq.heappop(ready)
                start = max(t, r)
                finish = start + w
                machines[free] = finish
                if finish > d + EPS:
                    misses += 1
                continue  # keep dispatching at the current instant
            # every machine busy: advance to the next event
            nxt = min(machines)
            if i < len(pending):
                nxt = min(nxt, pending[i][0])
            t = max(nxt, t + EPS)
            continue
        # nothing ready: jump to the next release
        t = max(pending[i][0], t)
    return misses


def main():
    bounds, relaxed = [], []
    for seed in range(N_SEEDS):
        jobs = gen_jobs(seed)
        b = capacity_bound(jobs)
        e = relaxed_edf_misses(jobs)
        bounds.append(b)
        relaxed.append(e)
        print(f"seed {seed:2d}: capacity bound {b:3d} misses "
              f"({100*b/N_JOBS:.2f}%), relaxed EDF {e:3d} ({100*e/N_JOBS:.2f}%)",
              flush=True)

    out = {
        "description": "Capacity lower bound and clairvoyant relaxed EDF, "
                       "hectic day, C_max=5, expected-service instance",
        "n_seeds": N_SEEDS,
        "capacity_bound_misses": bounds,
        "capacity_bound_miss_pct_mean": round(100 * float(np.mean(bounds)) / N_JOBS, 4),
        "relaxed_edf_misses": relaxed,
        "relaxed_edf_miss_pct_mean": round(100 * float(np.mean(relaxed)) / N_JOBS, 4),
    }
    out_path = Path(__file__).parent / "Outputs" / "phase3" / "capacity_bound.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"mean capacity bound: {out['capacity_bound_miss_pct_mean']}%  "
          f"mean relaxed EDF: {out['relaxed_edf_miss_pct_mean']}%")


if __name__ == "__main__":
    main()
