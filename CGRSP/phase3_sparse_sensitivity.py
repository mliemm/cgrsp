"""Vary only the three estimated service cells and measure dispatch exposure.

The mean factors 0.75 and 1.25 and dispersion factors 0.5 and 2 are modelling
stress brackets, not confidence bounds on the unobserved cells. All other
execution cells, prices, arrivals and availability assumptions remain fixed.
"""

import _repro  # noqa: F401
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats

from cgrsp.data.execution_times import (
    EXECUTION_TIME_LOOKUP, EXECUTION_TIME_SIGMA_LN, POOLED_SIGMA_LN, SAMPLE_SIZES,
)
import cgrsp.schedulers.rolling_horizon_scheduler as rh
from phase3_statistical import SCHEDULER_MAP, _run_one

N_SEEDS = 10
POLICIES = ["FIFO", "SPT", "EDF", "SPT+Rescue", "RH", "CADR"]
ARMS = {"baseline": (1, 1), "mean_075": (.75, 1), "mean_125": (1.25, 1),
        "sigma_050": (1, .5), "sigma_200": (1, 2)}
SPARSE_CELLS = [key for key, count in SAMPLE_SIZES.items() if count < 5]


@contextmanager
def sparse_calibration(mean_factor, sigma_factor):
    """Apply one calibration arm and restore the tables even after a failure."""
    means = EXECUTION_TIME_LOOKUP.copy()
    sigmas = EXECUTION_TIME_SIGMA_LN.copy()
    service_mean = rh._MEAN_SERVICE_TIME
    try:
        for key in SPARSE_CELLS:
            EXECUTION_TIME_LOOKUP[key] = means[key] * mean_factor
            EXECUTION_TIME_SIGMA_LN[key] = sigmas.get(key, POOLED_SIGMA_LN) * sigma_factor
        rh._MEAN_SERVICE_TIME = rh._mean_service_time()
        yield
    finally:
        EXECUTION_TIME_LOOKUP.clear()
        EXECUTION_TIME_LOOKUP.update(means)
        EXECUTION_TIME_SIGMA_LN.clear()
        EXECUTION_TIME_SIGMA_LN.update(sigmas)
        rh._MEAN_SERVICE_TIME = service_mean


def main():
    started = time.monotonic()
    arms = {}
    for arm, (mean_factor, sigma_factor) in ARMS.items():
        rows = {}
        with sparse_calibration(mean_factor, sigma_factor):
            for name in POLICIES:
                runs = [_run_one(SCHEDULER_MAP[name], seed) for seed in range(N_SEEDS)]
                assert all(run["completed"] == 950 for run in runs)
                rows[name] = {
                    "wait_min": float(np.mean([run["avg_wait_min"] for run in runs])),
                    "miss_pct": float(np.mean([run["deadline_miss_pct"] for run in runs])),
                    "tard_min": float(np.mean([run["avg_tardiness_min"] for run in runs])),
                    "sparse_dispatch_pct": float(np.mean([run["sparse_dispatch_pct"] for run in runs])),
                    "raw_miss_pct": [run["deadline_miss_pct"] for run in runs],
                }
                print(arm, name, rows[name]["miss_pct"], flush=True)
        test = stats.ttest_rel(rows["RH"]["raw_miss_pct"], rows["CADR"]["raw_miss_pct"])
        arms[arm] = {"mean_factor": mean_factor, "sigma_factor": sigma_factor,
                     "schedulers": rows,
                     "rh_vs_cadr_miss_p": float(test.pvalue)}
    output = {
        "metadata": {"n_seeds": N_SEEDS, "n_jobs": 950, "concurrency": 5,
                     "scope": "Six focal policies; factors are stress brackets, not calibration confidence intervals.",
                     "sparse_cells": [{"complexity": int(key[0]), "gpu": key[1].value,
                                       "samples": SAMPLE_SIZES[key]} for key in SPARSE_CELLS],
                     "elapsed_sec": round(time.monotonic() - started, 1)},
        "arms": arms,
    }
    target = Path(__file__).parent / "Outputs/phase3/sparse_sensitivity.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
