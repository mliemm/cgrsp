"""Two assumptions the calibration cannot rule out, bracketed on the hectic day.

Arm group 1, label noise: schedulers plan with an observed complexity label that
differs from the realised one. In the dataset the label is an ex-post render-time
tercile, so a deployed classifier would be imperfect (pixel count alone has
correlation -0.054 with render time). ``adj_*`` arms move a job to a neighbouring
class with the stated probability; ``uninformative`` draws the label from the
class mix independently of the job. Realised execution always uses the true class.

Arm group 2, GPU speed: the A4000 stratum means are not scene-matched. After the concurrency exclusion, on the 2
scenes it shares with the A5000 its geometric-mean render time is 1.165 times the
A5000's, so every A4000 cell is scaled to W(3, A4000) = 1.165 * W(3, A5000) (indicative only: two scenes).

Six focal policies, 10 seeds, 950 jobs, C_max = 5. Factors are stress brackets,
not estimates.
"""

import _repro  # noqa: F401
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy import stats

from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, GPUType
from cgrsp.data.job_generator import COMPLEXITY_DISTRIBUTION
import cgrsp.schedulers.rolling_horizon_scheduler as rh
from cgrsp.simulator.core import Simulator
from phase3_statistical import SCHEDULER_MAP, _run_one

N_SEEDS = 10
POLICIES = ["FIFO", "SPT", "EDF", "SPT+Rescue", "RH", "CADR"]
SCENE_MATCHED_RATIO = 1.165  # A4000 / A5000 on the 2 shared scenes left after the concurrency exclusion
ARMS = {
    "baseline": {},
    "adj_20": {"label_error": 0.20},
    "adj_40": {"label_error": 0.40},
    "uninformative": {"label_error": "uninformative"},
    "a4000_scene_matched": {"a4000_ratio": SCENE_MATCHED_RATIO},
}


def _observed(true, mode, rng):
    if mode == "uninformative":
        classes, weights = zip(*sorted(COMPLEXITY_DISTRIBUTION.items()))
        return rng.choices(classes, weights=weights)[0]
    if rng.random() >= mode:
        return true
    if true == 1:
        return 2
    if true == 3:
        return 2
    return rng.choice([1, 3])


@contextmanager
def arm(spec, seed):
    """Apply one arm, restoring every patched table and method afterwards."""
    means = EXECUTION_TIME_LOOKUP.copy()
    service_mean = rh._MEAN_SERVICE_TIME
    original_submit = Simulator.submit_job
    try:
        if "a4000_ratio" in spec:
            scale = (spec["a4000_ratio"] * means[(3, GPUType.RTX_A5000)]
                     / means[(3, GPUType.RTX_A4000)])
            for key in means:
                if key[1] == GPUType.RTX_A4000:
                    EXECUTION_TIME_LOOKUP[key] = means[key] * scale
            rh._MEAN_SERVICE_TIME = rh._mean_service_time()
        if "label_error" in spec:
            rng = random.Random(10_007 * seed + 1)

            def submit(self, complexity, *args, **kwargs):
                job = original_submit(self, _observed(int(complexity), spec["label_error"], rng),
                                      *args, **kwargs)
                job.true_complexity = complexity
                return job

            Simulator.submit_job = submit
        yield
    finally:
        Simulator.submit_job = original_submit
        EXECUTION_TIME_LOOKUP.clear()
        EXECUTION_TIME_LOOKUP.update(means)
        rh._MEAN_SERVICE_TIME = service_mean


def main():
    started = time.monotonic()
    arms = {}
    for name, spec in ARMS.items():
        rows = {}
        for policy in POLICIES:
            runs = []
            for seed in range(N_SEEDS):
                with arm(spec, seed):
                    runs.append(_run_one(SCHEDULER_MAP[policy], seed))
            assert all(run["completed"] == 950 for run in runs)
            rows[policy] = {
                "wait_min": float(np.mean([r["avg_wait_min"] for r in runs])),
                "miss_pct": float(np.mean([r["deadline_miss_pct"] for r in runs])),
                "tard_min": float(np.mean([r["avg_tardiness_min"] for r in runs])),
                "cost_usd": float(np.mean([r["total_cost"] for r in runs])),
                "raw_miss_pct": [r["deadline_miss_pct"] for r in runs],
            }
            print(name, policy, round(rows[policy]["miss_pct"], 3), flush=True)
        test = stats.ttest_rel(rows["RH"]["raw_miss_pct"], rows["CADR"]["raw_miss_pct"])
        arms[name] = {"spec": spec, "schedulers": rows, "rh_vs_cadr_miss_p": float(test.pvalue)}
    output = {
        "metadata": {"n_seeds": N_SEEDS, "n_jobs": 950, "concurrency": 5,
                     "scene_matched_ratio": SCENE_MATCHED_RATIO,
                     "scope": "Six focal policies; arms are stress brackets, not estimates.",
                     "elapsed_sec": round(time.monotonic() - started, 1)},
        "arms": arms,
    }
    target = Path(__file__).parent / "Outputs/phase3/kappa_speed_sensitivity.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
