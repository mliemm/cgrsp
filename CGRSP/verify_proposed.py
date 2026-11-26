"""Run the constraint verifier on the proposed and hybrid schedulers (RH, CADR, Adaptive).

main.py verifies the seven baselines in the Phase 2 benchmark; this script applies the same
ConstraintVerifier to the three schedulers main.py does not run, on the identical seed-42
hectic-day workload, and writes Outputs/runpod_scheduler/verification_proposed.json.
"""
import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import json
from pathlib import Path

from main import run_experiment
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from phase3_statistical import SCHEDULER_MAP as SCHEDULERS

SEED = 42
PROPOSED = ["RH", "CADR", "Adaptive"]


def main():
    jobs = JobGenerator(JobGeneratorConfig(
        num_jobs=950, arrival_rate=0.100, seed=SEED,
        deadline_distribution=[(3600.0, 0.20), (28800.0, 0.80)],
    )).generate()
    report = {}
    for name in PROPOSED:
        _, verification = run_experiment(name, SCHEDULERS[name], jobs, 5, SEED)
        report[name] = verification
    out = Path(__file__).parent / "Outputs" / "runpod_scheduler" / "verification_proposed.json"
    out.write_text(json.dumps({"seed": SEED, "schedulers": report}, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
