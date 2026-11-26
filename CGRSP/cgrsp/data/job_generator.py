"""
CGRSP Job Generator
Generates rendering jobs with Poisson arrival process and empirical complexity distribution.

Based on the formulation:
  - Job arrivals follow Poisson process with rate λ
  - Complexity κ follows discrete distribution P(κ)
  - Deadline: d_j = t_j^submit + T_deadline, sampled from heterogeneous distribution
    (default: 20% tight 1h = 3600s, 80% loose 8h = 28800s)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

from .execution_times import GPUType, COMPLEXITY_LEVELS


# Complexity distribution from the sample sizes of the filtered calibration CSV
# (execution_times.py SAMPLE_SIZES: low=2052, medium=2127, high=2055 of 6,234,
# counted at the 65.2/73.9 s thresholds with strict less-than). Both the share of
# each class and the thresholds are near thirds by construction.
COMPLEXITY_DISTRIBUTION = {
    1: 0.3292,   # 32.92% low complexity
    2: 0.3412,   # 34.12% medium complexity
    3: 0.3296,   # 32.96% high complexity
}

# Default loose deadline (in seconds); main experiments use heterogeneous 20% tight (1h) / 80% loose (8h)
DEFAULT_DEADLINE_DURATION = 28800


class PriorityLevel(Enum):
    """Job priority levels for differentiated service."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Job:
    """
    Represents a single rendering job in the CGRSP system.

    Attributes:
        job_id: Unique job identifier
        complexity: Complexity level (1=Low, 2=Medium, 3=High)
        submit_time: Job submission timestamp
        deadline: Job deadline (submit_time + deadline_duration)
        priority: Service priority (low, medium, high)
        gpu_requirement: Number of GPUs required (default: 1)
        estimated_duration: Estimated execution time (optional)
    """
    job_id: int
    complexity: int
    submit_time: float
    deadline: float
    priority: str = PriorityLevel.MEDIUM.value
    gpu_requirement: int = 1
    estimated_duration: Optional[float] = None

    # Execution tracking (populated during simulation)
    assigned_gpu_type: Optional[GPUType] = None
    assigned_node_id: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    completed: bool = False
    deadline_met: Optional[bool] = None

    def __post_init__(self):
        """Validate job parameters."""
        if self.complexity not in COMPLEXITY_DISTRIBUTION:
            raise ValueError(
                f"Invalid complexity: {self.complexity}. "
                f"Must be one of {list(COMPLEXITY_DISTRIBUTION.keys())}"
            )

        if self.deadline <= self.submit_time:
            raise ValueError(
                f"Deadline ({self.deadline}) must be after submit time ({self.submit_time})"
            )

        if self.gpu_requirement < 1:
            raise ValueError(
                f"GPU requirement must be >= 1, got {self.gpu_requirement}"
            )

        if self.priority not in [p.value for p in PriorityLevel]:
            raise ValueError(
                f"Invalid priority: {self.priority}. "
                f"Must be one of {[p.value for p in PriorityLevel]}"
            )

    def get_waiting_time(self) -> float:
        """Calculate time job spent waiting in queue."""
        if self.start_time is None:
            return 0.0
        return self.start_time - self.submit_time

    def get_response_time(self) -> float:
        """Calculate total turnaround time (submit to finish)."""
        if self.finish_time is None:
            return 0.0
        return self.finish_time - self.submit_time

    def get_deadline_violation(self) -> float:
        """Calculate amount by which deadline was missed (0 if met)."""
        if self.finish_time is None:
            return 0.0
        return max(0.0, self.finish_time - self.deadline)

    def is_overdue(self, current_time: float) -> bool:
        """Check if job is past its deadline at current time."""
        return current_time > self.deadline

    def get_time_to_deadline(self, current_time: float) -> float:
        """Get remaining time until deadline."""
        return max(0.0, self.deadline - current_time)

    def get_execution_time(self, gpu_type: GPUType) -> float:
        """Get execution time for this job on a specific GPU type."""
        from .execution_times import get_execution_time
        return get_execution_time(self.complexity, gpu_type)


@dataclass
class JobGeneratorConfig:
    """Configuration for job generation."""
    num_jobs: int = 100
    arrival_rate: float = 0.1  # Jobs per second (λ)
    complexity_probs: Dict[int, float] = field(
        default_factory=lambda: COMPLEXITY_DISTRIBUTION.copy()
    )
    priority_probs: Dict[str, float] = field(
        default_factory=lambda: {
            PriorityLevel.LOW.value: 0.3,
            PriorityLevel.MEDIUM.value: 0.5,
            PriorityLevel.HIGH.value: 0.2
        }
    )
    deadline_duration: float = DEFAULT_DEADLINE_DURATION
    # Heterogeneous deadline windows: list of (duration_seconds, probability) pairs.
    # If provided, overrides deadline_duration.  Default: 20% tight (1 h) + 80% loose (8 h).
    deadline_distribution: Optional[List] = field(
        default_factory=lambda: [(3600.0, 0.20), (DEFAULT_DEADLINE_DURATION, 0.80)]
    )
    gpu_requirement: int = 1  # Most jobs require single GPU
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.num_jobs < 0:
            raise ValueError(f"num_jobs must be >= 0, got {self.num_jobs}")

        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {self.arrival_rate}")

        # Validate complexity probabilities sum to 1
        total_complexity_prob = sum(self.complexity_probs.values())
        if not np.isclose(total_complexity_prob, 1.0, atol=0.01):
            raise ValueError(
                f"Complexity probabilities must sum to 1.0, "
                f"got {total_complexity_prob}"
            )

        # Validate priority probabilities sum to 1
        total_priority_prob = sum(self.priority_probs.values())
        if not np.isclose(total_priority_prob, 1.0, atol=0.01):
            raise ValueError(
                f"Priority probabilities must sum to 1.0, "
                f"got {total_priority_prob}"
            )

        if self.deadline_duration <= 0:
            raise ValueError(
                f"deadline_duration must be > 0, got {self.deadline_duration}"
            )

        # Build internal arrays for np.random.choice from deadline_distribution
        if self.deadline_distribution:
            durations, probs = zip(*self.deadline_distribution)
            if not np.isclose(sum(probs), 1.0, atol=0.01):
                raise ValueError("deadline_distribution probabilities must sum to 1.0")
            self._deadline_choices = list(durations)
            self._deadline_probs = list(probs)
        else:
            self._deadline_choices = [self.deadline_duration]
            self._deadline_probs = [1.0]


class JobGenerator:
    """
    Generates rendering jobs following Poisson arrival process.

    Job arrivals follow Poisson process with rate λ:
      - Inter-arrival times ~ Exponential(1/λ)
      - Number of arrivals in time t ~ Poisson(λ × t)

    Job complexity follows discrete distribution (see COMPLEXITY_DISTRIBUTION,
    the exact per-stratum proportions of the 6,234-job dataset):
      - P(κ=1) = 0.3292 (Low)
      - P(κ=2) = 0.3412 (Medium)
      - P(κ=3) = 0.3296 (High)

    Deadlines are sampled from a heterogeneous distribution (default: 20% tight 1h / 80% loose 8h).
    """

    def __init__(self, config: Optional[JobGeneratorConfig] = None):
        """
        Initialize job generator.

        Args:
            config: Job generation configuration. If None, uses defaults.
        """
        self.config = config or JobGeneratorConfig()

        # Set random seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.jobs: List[Job] = []
        self.next_job_id = 0

    def _next_arrival_time(self, current_time: float) -> float:
        """Sample the next arrival of the configured homogeneous process."""
        return current_time + np.random.exponential(1.0 / self.config.arrival_rate)

    def _make_job(self, submit_time: float) -> Job:
        """Sample job attributes and the shared deadline-window distribution."""
        complexity = np.random.choice(list(self.config.complexity_probs),
                                      p=list(self.config.complexity_probs.values()))
        priority = np.random.choice(list(self.config.priority_probs),
                                    p=list(self.config.priority_probs.values()))
        window = np.random.choice(self.config._deadline_choices,
                                  p=self.config._deadline_probs)
        job = Job(job_id=self.next_job_id, complexity=complexity,
                  submit_time=submit_time, deadline=submit_time + window,
                  priority=priority, gpu_requirement=self.config.gpu_requirement)
        self.next_job_id += 1
        return job

    def generate(self) -> List[Job]:
        """Generate a trace using this generator's arrival process."""
        self.jobs = []
        self.generate_batch(self.config.num_jobs)
        return self.jobs

    def generate_batch(self, batch_size: int, start_time: float = 0.0) -> List[Job]:
        """Append jobs after start_time, preserving the configured arrival process."""
        current_time = start_time
        batch = []
        for _ in range(batch_size):
            current_time = self._next_arrival_time(current_time)
            batch.append(self._make_job(current_time))
        self.jobs.extend(batch)
        return batch

    def get_statistics(self) -> Dict:
        """
        Get statistics about generated jobs.

        Returns:
            Dictionary with job statistics
        """
        if not self.jobs:
            return {}

        complexities = [job.complexity for job in self.jobs]
        priorities = [job.priority for job in self.jobs]
        submit_times = [job.submit_time for job in self.jobs]

        # Calculate inter-arrival times
        inter_arrivals = [
            submit_times[i] - submit_times[i-1]
            for i in range(1, len(submit_times))
        ]

        return {
            "total_jobs": len(self.jobs),
            "complexity_distribution": {
                "low": sum(1 for c in complexities if c == 1),
                "medium": sum(1 for c in complexities if c == 2),
                "high": sum(1 for c in complexities if c == 3),
            },
            "priority_distribution": {
                "low": sum(1 for p in priorities if p == PriorityLevel.LOW.value),
                "medium": sum(1 for p in priorities if p == PriorityLevel.MEDIUM.value),
                "high": sum(1 for p in priorities if p == PriorityLevel.HIGH.value),
            },
            "duration_hours": (submit_times[-1] - submit_times[0]) / 3600 if len(submit_times) > 1 else 0,
            "avg_inter_arrival_sec": np.mean(inter_arrivals) if inter_arrivals else 0,
            "first_job_time": submit_times[0] if submit_times else 0,
            "last_job_time": submit_times[-1] if submit_times else 0,
        }


class TimeVaryingJobGenerator(JobGenerator):
    """
    Job generator with time-varying arrival rate.

    Models realistic workload patterns with peak and off-peak hours.
    """

    def __init__(self, config: Optional[JobGeneratorConfig] = None,
                 arrival_rate_function: Optional[Callable[[float], float]] = None,
                 max_arrival_rate: Optional[float] = None):
        """
        Initialize time-varying job generator.

        Args:
            config: Job generation configuration
            arrival_rate_function: Function mapping time to arrival rate λ(t)
            max_arrival_rate: Positive global upper bound on a custom rate function
        """
        super().__init__(config)
        self.arrival_rate_function = arrival_rate_function or self._default_time_varying_rate
        if max_arrival_rate is None:
            if arrival_rate_function is not None:
                raise ValueError("A custom rate function requires max_arrival_rate")
            max_arrival_rate = 0.1 * 1.5
        if not np.isfinite(max_arrival_rate) or max_arrival_rate <= 0:
            raise ValueError("max_arrival_rate must be finite and positive")
        self.max_arrival_rate = max_arrival_rate

    @staticmethod
    def _default_time_varying_rate(current_time: float) -> float:
        """
        Default time-varying arrival rate.

        Models peak hours (9AM-5PM) with 1.5x base rate,
        off-peak with 0.7x base rate.
        """
        base_rate = 0.1  # jobs per second
        hour_of_day = (current_time / 3600) % 24

        if 9 <= hour_of_day <= 17:
            return base_rate * 1.5  # Business hours
        else:
            return base_rate * 0.7  # Off-hours

    def _next_arrival_time(self, current_time: float) -> float:
        """Thin a dominating Poisson process against the rate at candidate times.

        The caller supplies a global rate bound for custom functions. Zero-rate
        intervals are crossed by rejected candidates rather than by dividing by
        their instantaneous rate.
        """
        while True:
            current_time += np.random.exponential(1.0 / self.max_arrival_rate)
            rate = self.arrival_rate_function(current_time)
            if not np.isfinite(rate) or not 0 <= rate <= self.max_arrival_rate:
                raise ValueError("Arrival rate must lie between zero and max_arrival_rate")
            if np.random.random() < rate / self.max_arrival_rate:
                return current_time


if __name__ == "__main__":
    # Example usage
    print("CGRSP Job Generator")
    print("=" * 50)

    # Generate standard workload
    config = JobGeneratorConfig(
        num_jobs=100,
        arrival_rate=0.1,
        seed=42
    )

    generator = JobGenerator(config)
    jobs = generator.generate()

    stats = generator.get_statistics()
    print(f"\nGenerated Jobs:")
    print(f"  Total: {stats['total_jobs']}")
    print(f"  Complexity: {stats['complexity_distribution']}")
    print(f"  Priority: {stats['priority_distribution']}")
    print(f"  Duration: {stats['duration_hours']:.2f} hours")
    print(f"  Avg inter-arrival: {stats['avg_inter_arrival_sec']:.2f} sec")

    # Display first few jobs
    print(f"\nFirst 5 Jobs:")
    for job in jobs[:5]:
        time_to_deadline = job.deadline - job.submit_time
        print(f"  Job {job.job_id}: "
              f"κ={job.complexity}, "
              f"t={job.submit_time:.1f}s, "
              f"deadline={job.deadline:.1f}s, "
              f"window={time_to_deadline}s, "
              f"priority={job.priority}")
