"""
Phase 3 — Availability-Prior Robustness (rank permutation + RH prior ablation)
===============================================================================
Three arms. Arms A and B run on the hectic day (950 jobs, λ=0.100, C_max=5,
30 seeds, same job streams as the Phase 3 statistical benchmark); arm C runs on
both the hectic and the surge day.

Arm A (reversed β_g ranks): the availability model's baseline high-stock
probabilities are REVERSED across types (RTX 3090 becomes the best-stocked at
0.75, RTX A4000 the worst at 0.50) while RH's and Adaptive's fixed per-type
reliability priors keep their calibrated ordering. This is the adversarial
scenario for the priors: their rank is now maximally wrong. If the headline
scheduler ordering survives, the priors are not load-bearing.

Arm B (RH prior ablation): RH run with the per-type reliability prior zeroed
(stock-status penalty retained), standard availability. Compares against the
canonical RH baseline to isolate the prior's contribution.

Arm C (arrival-rate prior): every other entry point hands RH the generator's
exact λ, so the sensitivity of its anticipation gate to a mis-estimated arrival
rate is never measured. Here the true job stream is held fixed while RH is TOLD
an assumed λ at {0.5 … 4.0}× the truth. λ reaches nothing but _reserve_count,
which thresholds the offered load ρ = λ·E[S]/C against rho_gate, so the whole
estimate collapses to a binary reserve/do-not-reserve decision at a single
critical rate (reported as "lambda_gate" below). The response is therefore a
step, not a gradient: several multipliers are bit-identical and one crossing is
abrupt. On the hectic day the crossing sits between ×0.6 and ×0.75, so a
sufficiently large under-estimate turns reservation on where there is no spare
capacity to reserve; on the surge day it sits between ×2.0 and ×4.0, so only a
gross over-estimate forfeits the reservation benefit. A FIFO reference cell is
recorded per day so the degraded settings can be read against a
non-anticipating baseline. RH's use of λ is unchanged; only the value it is
given varies.

Output: Outputs/phase3/prior_robustness.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 and TZ=Asia/Taipei

import sys, os, json, time, inspect
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.data.runpod_availability import RunPodAvailabilityModel
from cgrsp.schedulers import (
    FIFOScheduler, SPTScheduler, EDFScheduler, LaxityRescueScheduler,
    RollingHorizonScheduler, CostAwareDeadlineRiskScheduler,
)
import cgrsp.schedulers.rolling_horizon_scheduler as rh_mod

SCHEDULER_MAP = {
    "FIFO":       lambda fm, seed, ar: FIFOScheduler(fm, seed=seed),
    "SPT":        lambda fm, seed, ar: SPTScheduler(fm, seed=seed),
    "EDF":        lambda fm, seed, ar: EDFScheduler(fm, seed=seed),
    "SPT+Rescue": lambda fm, seed, ar: LaxityRescueScheduler(fm, seed=seed),
    "RH":         lambda fm, seed, ar: RollingHorizonScheduler(
        fleet_manager=fm, seed=seed, arrival_rate=ar),
    "CADR":       lambda fm, seed, ar: CostAwareDeadlineRiskScheduler(
        fleet_manager=fm, seed=seed, cr_star=3.0, cost_aware=True),
}

CONCURRENCY = 5
N_SEEDS     = 30
DURATION    = 86400.0
N_JOBS      = 950
ARRIVAL     = 0.100
DEADLINES   = [(3600.0, 0.20), (28800.0, 0.80)]

# Reversed rank assignment: calibrated values are 3090 .50 / A5000 .65 /
# A4500 .70 / A4000 .75; the reversal swaps the best and worst ends.
REVERSED_BETA = {
    "RTX_3090": 0.75,
    "RTX_A5000": 0.70,
    "RTX_4090": 0.30,   # non-fleet type, unchanged
    "RTX_A4500": 0.65,
    "RTX_A4000": 0.50,
}

# Arm C: multipliers applied to the true λ to obtain the λ RH is told. 1.0 is
# the canonical setting every other experiment uses.
ARRIVAL_MULTIPLIERS = [0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0, 4.0]

# Same two day calibrations as the reservation sweep: hectic sits above the
# offered-load gate, surge below it.
ARRIVAL_DAYS = {
    "hectic": {"n_jobs": 950, "arrival_rate": 0.100},
    "surge":  {"n_jobs": 730, "arrival_rate": 0.020},
}


def _lambda_gate():
    """Critical assumed arrival rate at which RH switches reservation off.

    Read from the scheduler's own rho_gate default rather than restated here,
    so the reported cliff cannot drift away from the implementation. Solving
    rho_gate = λ·E[S]/C_max for λ gives the threshold.
    """
    rho_gate = inspect.signature(
        RollingHorizonScheduler.__init__).parameters["rho_gate"].default
    return rho_gate * CONCURRENCY / rh_mod._MEAN_SERVICE_TIME


def _run_one(sched_name, seed, beta_map=None, zero_prior=False,
             n_jobs=N_JOBS, arrival_rate=ARRIVAL, assumed_arrival_rate=None):
    """Run one seed.

    ``arrival_rate`` drives the job generator (the ground truth);
    ``assumed_arrival_rate`` is what the scheduler is told, defaulting to the
    truth so arms A and B are unaffected.
    """
    job_config = JobGeneratorConfig(
        num_jobs=n_jobs, arrival_rate=arrival_rate, seed=seed,
        deadline_distribution=DEADLINES,
    )
    jobs = JobGenerator(job_config).generate()

    cloud = RunPodCloudFleetManager(
        max_concurrency=CONCURRENCY,
        supported_types=ResourceType.supported_gpus(),
        random_seed=seed,
    )
    if beta_map is not None:
        cloud.availability_model.BASE_HIGH_STOCK_PROB = dict(beta_map)

    saved_prior = None
    if zero_prior:
        saved_prior = dict(rh_mod.GPU_RELIABILITY_PENALTY)
        for k in rh_mod.GPU_RELIABILITY_PENALTY:
            rh_mod.GPU_RELIABILITY_PENALTY[k] = 0.0
    try:
        told = arrival_rate if assumed_arrival_rate is None else assumed_arrival_rate
        sim = Simulator(fleet_manager=cloud, random_seed=seed)
        sim.set_scheduler(SCHEDULER_MAP[sched_name](cloud, seed, told))
        for job in jobs:
            sim.schedule_event(
                EventType.JOB_ARRIVAL, job.submit_time,
                {"complexity": job.complexity, "deadline": job.deadline,
                 "estimated_duration": job.estimated_duration or 100.0},
            )
        sim.run(duration=DURATION)
    finally:
        if saved_prior is not None:
            rh_mod.GPU_RELIABILITY_PENALTY.update(saved_prior)

    m = sim.get_metrics_summary()
    return {
        "wait_min": m["timing_metrics"]["avg_waiting_time"] / 60.0,
        "miss_pct": m["job_counts"]["deadline_miss_rate_pct"],
        "tard_min": float(m.get("avg_tardiness_s", 0.0)) / 60.0,
        "completed": m["job_counts"]["completed"],
    }


def _cell(runner):
    waits, misses, tards = [], [], []
    for seed in range(N_SEEDS):
        r = runner(seed)
        waits.append(r["wait_min"])
        misses.append(r["miss_pct"])
        tards.append(r["tard_min"])
    return {
        "wait_min_mean": round(float(np.mean(waits)), 4),
        "miss_pct_mean": round(float(np.mean(misses)), 4),
        "tard_min_mean": round(float(np.mean(tards)), 4),
        "wait_min_raw": [round(x, 4) for x in waits],
        "miss_pct_raw": [round(x, 4) for x in misses],
    }


def main():
    t0 = time.time()
    out = {"description": "Arm A: reversed beta_g ranks, six schedulers; "
                          "Arm B: RH with zeroed per-type reliability prior; "
                          "Arm C: RH told a mis-estimated arrival rate, swept "
                          "as a multiple of the true lambda on the hectic and "
                          "surge days. C_max=5, 30 seeds.",
           "reversed_beta": {k: v for k, v in REVERSED_BETA.items()
                             if k != "RTX_4090"},
           "arrival_multipliers": ARRIVAL_MULTIPLIERS,
           "arrival_days": ARRIVAL_DAYS,
           "lambda_gate": round(_lambda_gate(), 6),
           "arm_a_reversed": {}, "arm_b_rh_noprior": {},
           "arm_c_arrival_prior": {}}

    for name in SCHEDULER_MAP:
        out["arm_a_reversed"][name] = _cell(
            lambda seed, n=name: _run_one(n, seed, beta_map=REVERSED_BETA))
        c = out["arm_a_reversed"][name]
        print(f"armA {name:11s} miss {c['miss_pct_mean']:6.2f}%  "
              f"wait {c['wait_min_mean']:7.2f} min", flush=True)

    out["arm_b_rh_noprior"]["RH-noprior"] = _cell(
        lambda seed: _run_one("RH", seed, zero_prior=True))
    c = out["arm_b_rh_noprior"]["RH-noprior"]
    print(f"armB RH-noprior miss {c['miss_pct_mean']:6.2f}%  "
          f"wait {c['wait_min_mean']:7.2f} min", flush=True)

    # Arm C runs last so it cannot perturb arms A and B above.
    for day, p in ARRIVAL_DAYS.items():
        true_lambda, n_jobs = p["arrival_rate"], p["n_jobs"]
        cells = {}
        print(f"\narmC [{day}] true lambda={true_lambda}, "
              f"n_jobs={n_jobs}", flush=True)
        for mult in ARRIVAL_MULTIPLIERS:
            told = round(true_lambda * mult, 6)
            cell = _cell(lambda seed, a=told: _run_one(
                "RH", seed, n_jobs=n_jobs, arrival_rate=true_lambda,
                assumed_arrival_rate=a))
            cell["multiplier"] = mult
            cell["assumed_arrival_rate"] = told
            cell["true_arrival_rate"] = true_lambda
            cells[f"x{mult}"] = cell
            print(f"  x{mult:<4} told={told:<8.4f} miss {cell['miss_pct_mean']:6.2f}%"
                  f"  wait {cell['wait_min_mean']:7.2f} min"
                  f"  tard {cell['tard_min_mean']:7.2f} min", flush=True)
        # Non-anticipating reference for the same day and job streams.
        ref = _cell(lambda seed: _run_one(
            "FIFO", seed, n_jobs=n_jobs, arrival_rate=true_lambda))
        cells["FIFO_reference"] = ref
        print(f"  FIFO reference   miss {ref['miss_pct_mean']:6.2f}%"
              f"  wait {ref['wait_min_mean']:7.2f} min"
              f"  tard {ref['tard_min_mean']:7.2f} min", flush=True)
        out["arm_c_arrival_prior"][day] = cells

    out["runtime_s"] = round(time.time() - t0, 1)
    out_path = Path(__file__).parent / "Outputs" / "phase3" / "prior_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
