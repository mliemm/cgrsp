"""
CGRSP Main Runner - RunPod Cloud Simulation

Primary entry point for the Cloud GPU Rendering Scheduling Problem (CGRSP) simulator.
Strictly uses RunPod Secure Cloud pricing and qualitative availability (High/Medium/Low).
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import sys
import os
import json
import argparse
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.config import ResourceType
from cgrsp.data.execution_times import ExecutionTimeModel, GPUType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.evaluation.verification import ConstraintVerifier
from cgrsp.schedulers import (
    FIFOScheduler,
    SPTScheduler,
    EDFScheduler,
    LCFScheduler,
    BalancedScheduler,
    RandomScheduler,
    LaxityRescueScheduler,
)

def run_experiment(scheduler_name: str, scheduler_class, jobs: List, concurrency: int = 8, seed: int = 42):
    """
    Run a single experiment with a given scheduler.

    Returns:
        Tuple of (metrics summary, constraint verification report as a dict)
    """
    print(f"\n--- Running experiment with {scheduler_name} Scheduler ---")
    
    # 1. Setup Cloud Fleet Manager (No knowledge of stock)
    # Uses secure pricing and qualitative status (High/Medium/Low)
    cloud_fleet = RunPodCloudFleetManager(
        max_concurrency=concurrency,
        supported_types=[
            ResourceType.RTX_3090, 
            ResourceType.RTX_A5000,
            ResourceType.RTX_A4500,
            ResourceType.RTX_A4000
        ],
        random_seed=seed
    )

    # 2. Setup Simulator with Cloud Manager
    sim = Simulator(
        fleet_manager=cloud_fleet,
        random_seed=seed
    )

    # 3. Setup Scheduler
    scheduler = scheduler_class(fleet_manager=cloud_fleet, seed=seed)
    sim.set_scheduler(scheduler)

    # 4. Submit jobs via timed JOB_ARRIVAL events (proper Poisson timing)
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

    # 5. Run simulation (24h horizon)
    try:
        sim.run(duration=86400.0)
    except Exception as e:
        print(f"❌ Error during simulation for {scheduler_name}: {e}")
        traceback.print_exc()

    # 6. Verify constraints on the finished run.
    # Reported, never fatal: the numbers stay available for inspection either
    # way, and a verification failure is a finding rather than a crash.
    verification = ConstraintVerifier().verify_all(sim)
    if verification.all_passed and verification.total_violations == 0:
        print("   Verification: PASSED (0 constraint violations)")
    else:
        status = "PASSED" if verification.all_passed else "FAILED"
        print(f"   Verification: {status} ({verification.total_violations} violations)")
        for violation in verification.violations:
            print(f"      [{violation.severity}] {violation.constraint_name}"
                  f"/{violation.violation_type}: {violation.description}")

    # 7. Get Metrics
    return sim.get_metrics_summary(), verification.to_dict()

def main():
    parser = argparse.ArgumentParser(description="CGRSP RunPod Cloud Simulator")
    parser.add_argument("--jobs", type=int, default=950, help="Number of jobs (default: hectic day ~950)")
    parser.add_argument("--concurrency", type=int, default=5, help="User GPU quota (max concurrency, RunPod default=5)")
    parser.add_argument("--arrival-rate", type=float, default=0.100, help="Job arrival rate (jobs/sec, default: hectic day burst 0.100)")
    parser.add_argument("--output",
                        default=str(Path(__file__).parent / "Outputs" / "runpod_scheduler"),
                        help="Output directory (default: alongside this script, matching the phase scripts)")
    parser.add_argument("--pdf", default=None, help="Override output PDF path (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--schedulers", nargs="+", default=["FIFO", "SPT", "EDF", "LCF", "Balanced", "Random", "SPT+Rescue"],
                       help="Schedulers to compare")

    args = parser.parse_args()

    print("=" * 80)
    print("RUNPOD CLOUD SCHEDULER COMPARISON - MAIN RUNNER")
    print("=" * 80)
    print(f"\nScenario: {args.jobs} rendering jobs on RunPod Secure Cloud")
    print(f"User Quota (Max Concurrency): {args.concurrency} GPUs")
    print(f"Arrival Rate: {args.arrival_rate} jobs/sec")
    print("Pricing: RunPod Secure Cloud (March 2026)")
    print("Availability: Qualitative Stock Status (High/Medium/Low)")
    print("-" * 80)

    # Map names to classes
    scheduler_map = {
        "FIFO": FIFOScheduler,
        "SPT": SPTScheduler,
        "EDF": EDFScheduler,
        "LCF": LCFScheduler,
        "Balanced": BalancedScheduler,
        "Random": RandomScheduler,
        "SPT+Rescue": LaxityRescueScheduler,
    }

    # Generate Workload once
    job_config = JobGeneratorConfig(
        num_jobs=args.jobs,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        deadline_distribution=[(3600.0, 0.20), (28800.0, 0.80)],  # 20% tight 1h, 80% loose 8h
    )
    job_gen = JobGenerator(job_config)
    jobs = job_gen.generate()

    # Run Schedulers
    results = {}
    verification_reports = {}
    active_schedulers = [name for name in args.schedulers if name in scheduler_map]
    
    # Use shorter names for display headers
    display_names = {
        "FIFO": "FIFO",
        "SPT": "SPT",
        "EDF": "EDF",
        "LCF": "LCF",
        "Balanced": "BAL",
        "Random": "RND",
        "SPT+Rescue": "SPT+R",
    }

    for name in active_schedulers:
        results[name], verification_reports[name] = run_experiment(
            name, scheduler_map[name], jobs, args.concurrency, args.seed
        )

    # Print Results Comparison
    print("\n" + "=" * 80)
    print("CLOUD SIMULATION RESULTS COMPARISON")
    print("=" * 80)
    
    header = f"{'Metric':<25}"
    for name in active_schedulers:
        header += f" {display_names.get(name, name):>10}"
    print(header)
    print("-" * len(header))
    
    def print_metric(label, key1, key2, scale=1.0):
        print(f"{label:<25}", end="")
        for name in active_schedulers:
            val = results[name][key1][key2] * scale
            print(f" {val:>10.2f}", end="")
        print()

    print_metric("Avg Wait Time (min)", "timing_metrics", "avg_waiting_time", scale=1.0/60.0)
    print_metric("Total Cost ($)", "cost_metrics", "total_cost")
    print_metric("Fleet Utilization (%)", "utilization_legacy", "average", scale=100.0)
    
    print(f"{'Jobs Completed':<25}", end="")
    for name in active_schedulers:
        print(f" {results[name]['job_counts']['completed']:>10}", end="")
    print()

    print(f"{'Deadline Miss Rate (%)':<25}", end="")
    for name in active_schedulers:
        print(f" {results[name]['job_counts']['deadline_miss_rate_pct']:>10.2f}", end="")
    print()

    print(f"{'Constraint Violations':<25}", end="")
    for name in active_schedulers:
        print(f" {verification_reports[name]['total_violations']:>10}", end="")
    print()

    print("-" * len(header))

    total_violations = sum(r["total_violations"] for r in verification_reports.values())
    failed_schedulers = [n for n, r in verification_reports.items() if not r["all_passed"]]
    if failed_schedulers:
        print(f"⚠️  Constraint verification FAILED for: {', '.join(failed_schedulers)} "
              f"({total_violations} violations in total). Results are still reported above; "
              f"see 'verification' in report.json for detail.")
    elif total_violations:
        print(f"Constraint verification passed with {total_violations} warning-level findings "
              f"(see 'verification' in report.json).")
    else:
        print("Constraint verification passed for every scheduler (0 violations).")

    print("✅ Successfully completed all cloud simulations!")

    # Save full JSON report
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario": f"{args.jobs} rendering jobs on RunPod Secure Cloud",
            "num_jobs": args.jobs,
            "max_concurrency": args.concurrency,
            "arrival_rate_jobs_per_sec": args.arrival_rate,
            "simulation_duration_sec": 86400,
            "simulation_duration_hr": 24,
            "seed": args.seed,
            "availability_model": "Qualitative Stock Status (High/Medium/Low)",
            "pricing_source": "RunPod Secure Cloud (March 2026)",
            "schedulers_compared": active_schedulers,
        },
        "results": {
            name: results[name] for name in active_schedulers
        },
        "verification": {
            name: verification_reports[name] for name in active_schedulers
        }
    }

    report_path = output_path / "report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {report_path}")



if __name__ == "__main__":
    main()
