"""
CGRSP Execution Time Data
Empirical walltime function W(κ, g) from 6,234 real rendering jobs.

This module loads and processes real rendering performance data from
/data/rendering_performance_data.csv to derive empirical execution times.

Based on the formulation:
  - κ: Complexity level (1=Low, 2=Medium, 3=High)
  - g: GPU type (heterogeneous fleet)
  - W(κ, g): Execution time in seconds
  - C(g): Operational cost per second

Data Source:
    - 6,234 real rendering jobs from digitaldrape rendering logs (2025-07 to 2026-06; 6,030 production, 203 development-tagged, 1 untagged),
      after excluding the 393 jobs that ran while more than five jobs were in flight (scripts/filter_over_five_concurrency.py)
    - 4 GPU types: RTX 3090, RTX A5000, RTX A4500, RTX A4000
    - Complexity determined by render time percentiles
"""

import csv
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
from collections import defaultdict


class GPUType(Enum):
    """
    GPU types available in the CGRSP fleet.

    **CGRSP supports exactly these 4 GPU types with empirical data:**
    - RTX_A5000: Fastest (mean 72.6s), 4,680 jobs in dataset
    - RTX_3090: 2nd (mean 72.9s), 970 jobs in dataset
    - RTX_A4500: 3rd (mean 86.0s), 335 jobs in dataset
    - RTX_A4000: Slowest (mean 98.4s), 249 jobs in dataset

    Total: 6,234 real rendering jobs from the rendering pipeline (6,030 production, 203 development-tagged, 1 untagged).
    No other GPU types are supported or have empirical data.
    """
    RTX_3090 = "RTX_3090"
    RTX_A5000 = "RTX_A5000"
    RTX_A4500 = "RTX_A4500"
    RTX_A4000 = "RTX_A4000"

    @classmethod
    def supported_gpus(cls) -> list:
        """Return list of supported GPU types."""
        return [cls.RTX_3090, cls.RTX_A5000, cls.RTX_A4500, cls.RTX_A4000]

    @classmethod
    def is_supported(cls, gpu_type: str) -> bool:
        """Check if a GPU type string is supported."""
        return gpu_type in [g.value for g in cls.supported_gpus()]


# GPU name mapping from CSV to enum
GPU_NAME_MAP = {
    'NVIDIA GeForce RTX 3090': GPUType.RTX_3090,
    'NVIDIA RTX A5000': GPUType.RTX_A5000,
    'NVIDIA RTX A4500': GPUType.RTX_A4500,
    'NVIDIA RTX A4000': GPUType.RTX_A4000,
}

# Supported GPU types (explicit definition for validation)
SUPPORTED_GPU_TYPES = {
    'RTX_3090',
    'RTX_A5000',
    'RTX_A4500',
    'RTX_A4000',
}


# Complexity levels based on render time percentiles from real data
# Thresholds derived from 6,234 real rendering jobs
# Low: < 65.2s (33rd percentile)
# Medium: 65.2s - 73.9s (33rd-67th percentile)
# High: >= 73.9s (67th percentile)
COMPLEXITY_LEVELS = {
    1: "Low (<65.2s)",
    2: "Medium (65.2-73.9s)",
    3: "High (>=73.9s)"
}

# Complexity thresholds (in seconds)
COMPLEXITY_THRESHOLDS = {
    'low_threshold': 65.2,    # 33rd percentile
    'high_threshold': 73.9,   # 67th percentile
}


# Empirical execution times (seconds) from 6,234 real rendering jobs
# (production digitaldrape logs, 2025-07 to 2026-06, jobs that ran while more than
# five jobs were in flight excluded). Mean render time per (complexity, GPU)
# stratum. Real means are used wherever n >= 5; the three sparse cells (no
# low-complexity A4000/A4500 jobs; A4000 medium has n=2) use a relative-performance
# estimate anchored on the same GPU's nearest dense cell.
EXECUTION_TIME_LOOKUP: Dict[Tuple[int, GPUType], float] = {
    # Complexity 1 (Low): render time < 65.2 s
    (1, GPUType.RTX_3090): 59.5,    # n=368
    (1, GPUType.RTX_A5000): 60.7,   # n=1684
    (1, GPUType.RTX_A4500): 62.0,   # estimated (no low-complexity A4500 jobs)
    (1, GPUType.RTX_A4000): 63.3,   # estimated (no low-complexity A4000 jobs)

    # Complexity 2 (Medium): 65.2-73.9 s
    (2, GPUType.RTX_3090): 69.6,    # n=282
    (2, GPUType.RTX_A5000): 69.1,   # n=1799
    (2, GPUType.RTX_A4500): 70.6,   # n=44
    (2, GPUType.RTX_A4000): 72.1,   # estimated (n=2, below reliability threshold)

    # Complexity 3 (High): >= 73.9 s
    (3, GPUType.RTX_3090): 91.2,    # n=320
    (3, GPUType.RTX_A5000): 94.5,   # n=1197
    (3, GPUType.RTX_A4500): 88.3,   # n=291
    (3, GPUType.RTX_A4000): 98.6,   # n=247
}


# Within-stratum log-space dispersion sigma_ln for log-normal service times.
# Fitted directly from the 6,234-job dataset: sigma_ln = stdev(ln(render_time))
# within each (complexity, gpu) stratum that has sufficient samples (n >= 5).
# Realised execution time is sampled as W(j,g) ~ LogNormal(mu, sigma_ln) with
# mu = ln(E[W]) - sigma_ln^2 / 2 so that E[W] equals EXECUTION_TIME_LOOKUP[(k,g)].
# The three sparse cells (low-complexity A4500/A4000 with zero samples and
# medium-complexity A4000 with n=2) fall back to POOLED_SIGMA_LN, which is the
# n-weighted average of the nine dense-stratum sigma_ln values (0.105); the
# variance-pooled standard deviation of the same strata would be about 0.13.
# Frozen calibration constant: changing it would perturb every reported number.
POOLED_SIGMA_LN: float = 0.105

EXECUTION_TIME_SIGMA_LN: Dict[Tuple[int, GPUType], float] = {
    (1, GPUType.RTX_3090): 0.060,
    (1, GPUType.RTX_A5000): 0.074,

    (2, GPUType.RTX_3090): 0.036,
    (2, GPUType.RTX_A5000): 0.035,
    (2, GPUType.RTX_A4500): 0.029,

    (3, GPUType.RTX_3090): 0.200,
    (3, GPUType.RTX_A5000): 0.210,
    (3, GPUType.RTX_A4500): 0.207,
    (3, GPUType.RTX_A4000): 0.234,
}


# Sample sizes for each (complexity, gpu) combination from real data.
# Recounted from the filtered CSV at the 65.2 s / 73.9 s thresholds with strict
# less-than (2,052 / 2,127 / 2,055 overall). Not read by the simulator at run time.
SAMPLE_SIZES: Dict[Tuple[int, GPUType], int] = {
    (1, GPUType.RTX_3090): 368,
    (1, GPUType.RTX_A5000): 1684,
    (1, GPUType.RTX_A4500): 0,   # No low-complexity A4500 jobs
    (1, GPUType.RTX_A4000): 0,   # No low-complexity A4000 jobs

    (2, GPUType.RTX_3090): 282,
    (2, GPUType.RTX_A5000): 1799,
    (2, GPUType.RTX_A4500): 44,
    (2, GPUType.RTX_A4000): 2,

    (3, GPUType.RTX_3090): 320,
    (3, GPUType.RTX_A5000): 1197,
    (3, GPUType.RTX_A4500): 291,
    (3, GPUType.RTX_A4000): 247,
}


# Operational pricing (cost per second) from real 2026 data
# Values from distributions_20260310_234905.json (securePrice)
# RTX 3090: $0.46/hr
# RTX A5000: $0.27/hr
# RTX A4500: $0.25/hr
# RTX A4000: $0.25/hr
PRICING_LOOKUP: Dict[GPUType, float] = {
    GPUType.RTX_3090: 0.46 / 3600.0,
    GPUType.RTX_A5000: 0.27 / 3600.0,
    GPUType.RTX_A4500: 0.25 / 3600.0,
    GPUType.RTX_A4000: 0.25 / 3600.0,
}


# GPU specifications for reference
# performance_rank and relative_performance are empirical, from the per-GPU
# mean render time over the 6,234-job dataset (A5000 72.59s, 3090 72.90s,
# A4500 85.97s, A4000 98.35s). relative_performance is normalised to the
# fastest GPU (A5000 = 1.00). cuda_cores and memory_gb are hardware specs.
GPU_SPECIFICATIONS = {
    GPUType.RTX_3090: {
        "cuda_cores": 10496,
        "memory_gb": 24,
        "performance_rank": 2,
        "relative_performance": 1.00,
        "jobs_in_dataset": 970,
    },
    GPUType.RTX_A5000: {
        "cuda_cores": 8192,
        "memory_gb": 24,
        "performance_rank": 1,
        "relative_performance": 1.00,  # Fastest (baseline)
        "jobs_in_dataset": 4680,
    },
    GPUType.RTX_A4500: {
        "cuda_cores": 7168,
        "memory_gb": 20,
        "performance_rank": 3,
        "relative_performance": 0.84,
        "jobs_in_dataset": 335,
    },
    GPUType.RTX_A4000: {
        "cuda_cores": 6144,
        "memory_gb": 16,
        "performance_rank": 4,
        "relative_performance": 0.74,
        "jobs_in_dataset": 249,
    },
}


def load_rendering_data(data_path: Optional[str] = None) -> Dict:
    """
    Load rendering performance data from CSV file.

    Args:
        data_path: Path to rendering_performance_data.csv.
                   If None, uses default path relative to project root.

    Returns:
        Dictionary with processed statistics
    """
    if data_path is None:
        # Try to find the data file relative to this module
        module_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        data_path = os.path.join(project_root, 'data', 'rendering_performance_data.csv')

    if not os.path.exists(data_path):
        # Return pre-computed statistics if data file not found
        return get_precomputed_stats()

    data = []
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Group by GPU and complexity
    gpu_times = defaultdict(list)
    for row in data:
        gpu = row['gpu_used']
        render_time = float(row['render_time_s'])
        gpu_short = GPU_NAME_MAP.get(gpu)
        if gpu_short:
            gpu_times[gpu_short].append(render_time)

    return {
        'total_jobs': len(data),
        'gpu_times': dict(gpu_times),
        'source': data_path,
    }


def get_precomputed_stats() -> Dict:
    """Return precomputed statistics from the 6,234 job dataset."""
    return {
        'total_jobs': 6234,
        'gpu_times': {
            GPUType.RTX_3090: [72.90] * 970,  # Mean render time from real data
            GPUType.RTX_A5000: [72.59] * 4680,
            GPUType.RTX_A4500: [85.97] * 335,
            GPUType.RTX_A4000: [98.35] * 249,
        },
        'source': 'precomputed',
    }


def get_complexity_from_render_time(render_time: float) -> int:
    """
    Determine complexity level from render time using percentile thresholds.

    Args:
        render_time: Render time in seconds

    Returns:
        Complexity level (1=Low, 2=Medium, 3=High)
    """
    if render_time < COMPLEXITY_THRESHOLDS['low_threshold']:
        return 1
    elif render_time < COMPLEXITY_THRESHOLDS['high_threshold']:
        return 2
    else:
        return 3


@dataclass
class ExecutionTimeModel:
    """
    Model for execution time prediction based on empirical data.

    Implements the walltime function W(κ, g) from the CGRSP formulation.
    """

    @staticmethod
    def validate() -> bool:
        """
        Validate that all required execution times are present and only supported GPUs are used.

        Validates:
        1. All 4 supported GPU types are present
        2. All 3 complexity levels are covered
        3. Complete 4x3 coverage (4 GPUs × 3 complexities = 12 combinations)

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        # Check that we have exactly the 4 supported GPU types
        supported_gpus = GPUType.supported_gpus()
        if len(supported_gpus) != 4:
            raise ValueError(
                f"Expected exactly 4 supported GPU types, found {len(supported_gpus)}"
            )

        # Check that all 4 GPUs have data for all 3 complexities
        required_combinations = [
            (complexity, gpu_type)
            for complexity in [1, 2, 3]
            for gpu_type in supported_gpus
        ]

        missing = [
            combo for combo in required_combinations
            if combo not in EXECUTION_TIME_LOOKUP
        ]

        if missing:
            raise ValueError(
                f"Missing execution times for combinations: {missing}. "
                f"CGRSP requires complete 4x3 coverage (4 GPUs × 3 complexities)."
            )

        # Validate no unsupported GPU types in the lookup
        unsupported_gpus = set(
            gpu_type for (_, gpu_type) in EXECUTION_TIME_LOOKUP.keys()
        ) - set(supported_gpus)

        if unsupported_gpus:
            raise ValueError(
                f"Found unsupported GPU types in EXECUTION_TIME_LOOKUP: {unsupported_gpus}. "
                f"CGRSP only supports: {[g.value for g in supported_gpus]}"
            )

        return True

    @staticmethod
    def get_execution_time(complexity: int, gpu_type: GPUType) -> float:
        """
        Get execution time W(κ, g) for a given complexity and GPU type.

        Based on 6,234 real rendering jobs from the rendering pipeline (6,030 production, 203 development-tagged, 1 untagged).

        Args:
            complexity: Job complexity level (1, 2, or 3)
            gpu_type: GPU type from the fleet

        Returns:
            Execution time in seconds

        Raises:
            ValueError: If complexity or GPU type is invalid
        """
        if complexity not in [1, 2, 3]:
            raise ValueError(
                f"Invalid complexity level: {complexity}. "
                f"Must be 1, 2, or 3."
            )

        if not isinstance(gpu_type, GPUType):
            raise ValueError(
                f"Invalid GPU type: {gpu_type}. "
                f"Must be a GPUType enum value."
            )

        key = (complexity, gpu_type)
        if key not in EXECUTION_TIME_LOOKUP:
            raise ValueError(
                f"No execution time data for complexity={complexity}, "
                f"gpu_type={gpu_type}"
            )

        return EXECUTION_TIME_LOOKUP[key]

    @staticmethod
    def sample_execution_time(
        complexity: int,
        gpu_type: GPUType,
        rng: "random.Random",
        sigma_ln: Optional[float] = None,
    ) -> float:
        """
        Sample a realised execution time from the log-normal service model.

        W(j,g) ~ LogNormal(mu, sigma_ln) with mu = ln(E[W]) - sigma_ln^2 / 2,
        so the expected value equals the empirical stratum mean
        EXECUTION_TIME_LOOKUP[(complexity, gpu_type)]. This keeps the
        deterministic mean-based results as the expected case while reflecting
        the empirical within-stratum dispersion of the 6,234-job dataset.

        Args:
            complexity: Job complexity level (1, 2, or 3).
            gpu_type: GPU type the job is dispatched to.
            rng: A seeded random.Random instance for reproducibility.
            sigma_ln: Optional override for the log-space std. If None, the
                fitted per-stratum value is used, falling back to the pooled
                within-stratum value for strata without a reliable fit.

        Returns:
            A strictly positive realised execution time in seconds.
        """
        mean = ExecutionTimeModel.get_execution_time(complexity, gpu_type)
        if sigma_ln is None:
            sigma_ln = EXECUTION_TIME_SIGMA_LN.get(
                (complexity, gpu_type), POOLED_SIGMA_LN
            )
        if sigma_ln <= 0.0:
            return mean
        import math

        mu = math.log(mean) - 0.5 * sigma_ln * sigma_ln
        return rng.lognormvariate(mu, sigma_ln)

    @staticmethod
    def get_sample_size(complexity: int, gpu_type: GPUType) -> int:
        """
        Get the number of real jobs in the dataset for this combination.

        Args:
            complexity: Job complexity level (1, 2, or 3)
            gpu_type: GPU type from the fleet

        Returns:
            Number of jobs in the empirical dataset
        """
        return SAMPLE_SIZES.get((complexity, gpu_type), 0)

    @staticmethod
    def get_pricing(gpu_type: GPUType) -> float:
        """
        Get operational cost per second C(g) for a GPU type.

        Args:
            gpu_type: GPU type from the fleet

        Returns:
            Cost per second in USD
        """
        if not isinstance(gpu_type, GPUType):
            raise ValueError(
                f"Invalid GPU type: {gpu_type}. "
                f"Must be a GPUType enum value."
            )

        return PRICING_LOOKUP[gpu_type]

    @staticmethod
    def get_job_cost(complexity: int, gpu_type: GPUType) -> float:
        """
        Calculate total operational cost for a job.

        Cost = W(κ, g) × C(g)

        Args:
            complexity: Job complexity level (1, 2, or 3)
            gpu_type: GPU type executing the job

        Returns:
            Total cost in USD
        """
        execution_time = ExecutionTimeModel.get_execution_time(
            complexity, gpu_type
        )
        pricing = ExecutionTimeModel.get_pricing(gpu_type)
        return execution_time * pricing

    @staticmethod
    def get_all_execution_times() -> Dict[Tuple[int, GPUType], float]:
        """Get complete execution time lookup table."""
        return EXECUTION_TIME_LOOKUP.copy()

    @staticmethod
    def get_execution_times_by_complexity(complexity: int) -> Dict[GPUType, float]:
        """
        Get execution times for all GPU types at a given complexity level.

        Args:
            complexity: Job complexity level (1, 2, or 3)

        Returns:
            Dictionary mapping GPU types to execution times
        """
        if complexity not in [1, 2, 3]:
            raise ValueError(
                f"Invalid complexity level: {complexity}. "
                f"Must be 1, 2, or 3."
            )

        return {
            gpu_type: EXECUTION_TIME_LOOKUP[(complexity, gpu_type)]
            for gpu_type in GPUType
        }

    @staticmethod
    def get_execution_times_by_gpu(gpu_type: GPUType) -> Dict[int, float]:
        """
        Get execution times for all complexity levels on a given GPU type.

        Args:
            gpu_type: GPU type from the fleet

        Returns:
            Dictionary mapping complexity levels to execution times
        """
        if not isinstance(gpu_type, GPUType):
            raise ValueError(
                f"Invalid GPU type: {gpu_type}. "
                f"Must be a GPUType enum value."
            )

        return {
            complexity: EXECUTION_TIME_LOOKUP[(complexity, gpu_type)]
            for complexity in [1, 2, 3]
        }

    @staticmethod
    def get_gpu_specifications(gpu_type: Optional[GPUType] = None) -> Dict:
        """
        Get GPU specifications.

        Args:
            gpu_type: Specific GPU type, or None for all GPUs

        Returns:
            GPU specification(s)
        """
        if gpu_type is None:
            return {
                gpu_type.value: spec
                for gpu_type, spec in GPU_SPECIFICATIONS.items()
            }

        if not isinstance(gpu_type, GPUType):
            raise ValueError(
                f"Invalid GPU type: {gpu_type}. "
                f"Must be a GPUType enum value."
            )

        return GPU_SPECIFICATIONS[gpu_type]

    @staticmethod
    def calculate_speedup(
        from_gpu: GPUType,
        to_gpu: GPUType,
        complexity: int
    ) -> float:
        """
        Calculate speedup ratio when switching from one GPU to another.

        Speedup = W(κ, from_gpu) / W(κ, to_gpu)

        Args:
            from_gpu: Source GPU type
            to_gpu: Target GPU type
            complexity: Job complexity level

        Returns:
            Speedup ratio (>1 means to_gpu is faster)
        """
        time_from = ExecutionTimeModel.get_execution_time(complexity, from_gpu)
        time_to = ExecutionTimeModel.get_execution_time(complexity, to_gpu)
        return time_from / time_to


# Convenience functions for backward compatibility
def get_execution_time(complexity: int, gpu_type: GPUType) -> float:
    """Get execution time W(κ, g) for a given complexity and GPU type."""
    return ExecutionTimeModel.get_execution_time(complexity, gpu_type)


def get_pricing(gpu_type: GPUType) -> float:
    """Get operational cost per second C(g) for a GPU type."""
    return ExecutionTimeModel.get_pricing(gpu_type)


def get_job_cost(complexity: int, gpu_type: GPUType) -> float:
    """Calculate total operational cost for a job."""
    return ExecutionTimeModel.get_job_cost(complexity, gpu_type)


# Validate on module import
ExecutionTimeModel.validate()


if __name__ == "__main__":
    # Example usage and verification
    print("=" * 80)
    print("CGRSP Execution Time Model")
    print("Based on 6,234 real rendering jobs from the rendering pipeline (6,030 production, 203 development-tagged, 1 untagged)")
    print("=" * 80)

    # Display execution time lookup table
    print("\nExecution Time Lookup Table (W[κ, g] in seconds):")
    print("-" * 80)
    print(f"{'Complexity':<12} {'RTX_3090':<12} {'RTX_A5000':<12} {'RTX_A4500':<12} {'RTX_A4000':<12}")
    print("-" * 80)
    for complexity in [1, 2, 3]:
        row = [f"{complexity} ({COMPLEXITY_LEVELS[complexity][:15]})"]
        for gpu_type in GPUType:
            time_sec = EXECUTION_TIME_LOOKUP[(complexity, gpu_type)]
            cost = ExecutionTimeModel.get_job_cost(complexity, gpu_type)
            n = SAMPLE_SIZES.get((complexity, gpu_type), 0)
            row.append(f"{time_sec:6.1f}s (n={n:>3})")
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

    # Display pricing
    print("\n\nOperational Pricing (per second):")
    print("-" * 80)
    for gpu_type, price in PRICING_LOOKUP.items():
        print(f"  {gpu_type.value:12s}: ${price:.6f}/sec")

    # Display dataset statistics
    print("\n\nDataset Statistics:")
    print("-" * 80)
    stats = load_rendering_data()
    print(f"  Total jobs: {stats['total_jobs']}")
    print(f"  Data source: {stats['source']}")
    print("\n  Jobs by GPU type:")
    for gpu_type, specs in GPU_SPECIFICATIONS.items():
        print(f"    {gpu_type.value:12s}: {specs['jobs_in_dataset']} jobs")

    print("\n\nValidation:", "PASSED" if ExecutionTimeModel.validate() else "FAILED")
