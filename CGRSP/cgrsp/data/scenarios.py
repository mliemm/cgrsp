"""
CGRSP Workload Scenarios
Predefined workload configurations for different utilization levels and patterns.

Scenarios:
  - Light Load: 60% fleet utilization
  - Moderate Load: 85% fleet utilization
  - Heavy Load: 110% fleet utilization (oversubscription)
  - Peak Hour: Concentrated arrivals during business hours
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

from .job_generator import JobGenerator, JobGeneratorConfig, COMPLEXITY_DISTRIBUTION, DEFAULT_DEADLINE_DURATION
from .execution_times import GPUType, get_execution_time


class ScenarioType(Enum):
    """Types of workload scenarios."""
    LIGHT_LOAD = "light_load"
    MODERATE_LOAD = "moderate_load"
    HEAVY_LOAD = "heavy_load"
    PEAK_HOUR = "peak_hour"
    BURSTY = "bursty"
    CUSTOM = "custom"


@dataclass
class ScenarioConfig:
    """Configuration for a workload scenario."""
    scenario_type: ScenarioType
    num_jobs: int
    arrival_rate: float
    target_utilization: float  # Target fleet utilization (0.0 to 1.0+)
    description: str
    complexity_probs: Dict[int, float] = field(
        default_factory=lambda: COMPLEXITY_DISTRIBUTION.copy()
    )
    deadline_duration: float = DEFAULT_DEADLINE_DURATION
    peak_hours: Optional[List[tuple]] = None  # List of (start_hour, end_hour) tuples
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.num_jobs < 0:
            raise ValueError(f"num_jobs must be >= 0, got {self.num_jobs}")

        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {self.arrival_rate}")

        if not 0 < self.target_utilization:
            raise ValueError(
                f"target_utilization must be > 0, got {self.target_utilization}"
            )


# Predefined scenario configurations

# Light Load: 60% fleet utilization
# Suitable for testing and development
LIGHT_LOAD_CONFIG = ScenarioConfig(
    scenario_type=ScenarioType.LIGHT_LOAD,
    num_jobs=50,
    arrival_rate=0.05,
    target_utilization=0.6,
    description="Light workload - 60% fleet utilization. Suitable for testing."
)

# Moderate Load: 85% fleet utilization
# Standard production load with headroom for spikes
MODERATE_LOAD_CONFIG = ScenarioConfig(
    scenario_type=ScenarioType.MODERATE_LOAD,
    num_jobs=100,
    arrival_rate=0.1,
    target_utilization=0.85,
    description="Moderate workload - 85% fleet utilization. Standard production load."
)

# Heavy Load: 110% fleet utilization
# Stress testing with queue buildup
HEAVY_LOAD_CONFIG = ScenarioConfig(
    scenario_type=ScenarioType.HEAVY_LOAD,
    num_jobs=200,
    arrival_rate=0.2,
    target_utilization=1.1,
    description="Heavy workload - 110% fleet utilization. Oversubscribed stress test."
)

# Peak Hour: Concentrated arrivals during business hours
# Models realistic diurnal patterns
PEAK_HOUR_CONFIG = ScenarioConfig(
    scenario_type=ScenarioType.PEAK_HOUR,
    num_jobs=150,
    arrival_rate=0.15,
    target_utilization=0.95,
    description="Peak hour concentration - High arrival rate during 9AM-5PM.",
    peak_hours=[(9, 17)]  # 9AM to 5PM
)


class WorkloadScenario:
    """
    Generates workload scenarios for CGRSP simulation.

    Scenarios are calibrated to achieve target utilization levels
    based on empirical execution times and fleet configurations.
    """

    # Average execution time across all GPUs and complexities
    # Used for rough utilization estimation
    AVG_EXECUTION_TIME = 95.0  # seconds

    @staticmethod
    def calculate_arrival_rate_for_utilization(
        target_utilization: float,
        num_gpus: int,
        avg_execution_time: float = AVG_EXECUTION_TIME
    ) -> float:
        """
        Calculate arrival rate needed to achieve target utilization.

        Utilization = λ × N_g × avg_execution_time / total_gpus

        Solving for λ:
        λ = (target_utilization × total_gpus) / (num_gpus × avg_execution_time)

        Args:
            target_utilization: Target utilization (0.0 to 1.0+)
            num_gpus: GPUs per job (typically 1)
            avg_execution_time: Average job execution time

        Returns:
            Required arrival rate (jobs/second)
        """
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be > 0, got {num_gpus}")

        if avg_execution_time <= 0:
            raise ValueError(f"avg_execution_time must be > 0, got {avg_execution_time}")

        arrival_rate = (target_utilization * num_gpus) / avg_execution_time
        return arrival_rate

    @staticmethod
    def estimate_utilization(
        arrival_rate: float,
        num_gpus: int,
        avg_execution_time: float = AVG_EXECUTION_TIME
    ) -> float:
        """
        Estimate expected utilization for given arrival rate.

        Args:
            arrival_rate: Job arrival rate (jobs/second)
            num_gpus: Total number of GPUs in fleet
            avg_execution_time: Average job execution time

        Returns:
            Expected utilization (0.0 to 1.0+)
        """
        return (arrival_rate * avg_execution_time) / num_gpus

    @staticmethod
    def light_load(num_jobs: int = 50, seed: Optional[int] = None) -> JobGenerator:
        """
        Generate light load scenario (60% utilization).

        Args:
            num_jobs: Number of jobs to generate
            seed: Random seed for reproducibility

        Returns:
            Configured job generator
        """
        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=LIGHT_LOAD_CONFIG.arrival_rate,
            complexity_probs=LIGHT_LOAD_CONFIG.complexity_probs,
            deadline_duration=LIGHT_LOAD_CONFIG.deadline_duration,
            seed=seed
        )

        return JobGenerator(config)

    @staticmethod
    def moderate_load(num_jobs: int = 100, seed: Optional[int] = None) -> JobGenerator:
        """
        Generate moderate load scenario (85% utilization).

        Args:
            num_jobs: Number of jobs to generate
            seed: Random seed for reproducibility

        Returns:
            Configured job generator
        """
        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=MODERATE_LOAD_CONFIG.arrival_rate,
            complexity_probs=MODERATE_LOAD_CONFIG.complexity_probs,
            deadline_duration=MODERATE_LOAD_CONFIG.deadline_duration,
            seed=seed
        )

        return JobGenerator(config)

    @staticmethod
    def heavy_load(num_jobs: int = 200, seed: Optional[int] = None) -> JobGenerator:
        """
        Generate heavy load scenario (110% utilization).

        Args:
            num_jobs: Number of jobs to generate
            seed: Random seed for reproducibility

        Returns:
            Configured job generator
        """
        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=HEAVY_LOAD_CONFIG.arrival_rate,
            complexity_probs=HEAVY_LOAD_CONFIG.complexity_probs,
            deadline_duration=HEAVY_LOAD_CONFIG.deadline_duration,
            seed=seed
        )

        return JobGenerator(config)

    @staticmethod
    def peak_hour(num_jobs: int = 150, seed: Optional[int] = None) -> JobGenerator:
        """
        Generate peak hour scenario with concentrated arrivals.

        Models realistic workload patterns where jobs arrive more frequently
        during business hours (9AM-5PM).

        Args:
            num_jobs: Number of jobs to generate
            seed: Random seed for reproducibility

        Returns:
            Configured time-varying job generator
        """
        from .job_generator import TimeVaryingJobGenerator

        def peak_hour_rate(current_time: float) -> float:
            """Time-varying arrival rate with peak hours."""
            base_rate = 0.08  # Base rate
            peak_rate = 0.2   # Peak rate (2.5x base)

            hour_of_day = (current_time / 3600) % 24

            # Peak hours: 9AM-5PM
            if 9 <= hour_of_day <= 17:
                return peak_rate
            else:
                return base_rate

        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=0.15,  # Average rate
            complexity_probs=PEAK_HOUR_CONFIG.complexity_probs,
            deadline_duration=PEAK_HOUR_CONFIG.deadline_duration,
            seed=seed
        )

        return TimeVaryingJobGenerator(config, arrival_rate_function=peak_hour_rate,
                                      max_arrival_rate=0.2)

    @staticmethod
    def bursty_load(
        num_jobs: int = 100,
        num_bursts: int = 5,
        seed: Optional[int] = None
    ) -> JobGenerator:
        """
        Generate bursty workload with periodic job clusters.

        Models scenarios where jobs arrive in bursts (e.g., batch submissions).

        Args:
            num_jobs: Total number of jobs to generate
            num_bursts: Number of burst periods
            seed: Random seed for reproducibility

        Returns:
            Job generator with bursty pattern
        """
        # Create custom generator for bursty pattern
        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=0.1,
            deadline_duration=DEFAULT_DEADLINE_DURATION,
            seed=seed
        )

        generator = JobGenerator(config)

        # Override generate method for bursty pattern
        original_generate = generator.generate

        def bursty_generate():
            """Generate jobs in bursts."""
            import numpy as np

            jobs = []
            current_time = 0.0
            jobs_per_burst = num_jobs // num_bursts

            for burst in range(num_bursts):
                # Burst period: high arrival rate
                for _ in range(jobs_per_burst):
                    inter_arrival = np.random.exponential(1.0 / 0.5)  # High rate
                    current_time += inter_arrival

                    # Sample complexity and priority
                    complexity = np.random.choice(
                        [1, 2, 3],
                        p=list(config.complexity_probs.values())
                    )
                    priority = np.random.choice(
                        ["low", "medium", "high"],
                        p=[0.3, 0.5, 0.2]
                    )

                    from .job_generator import Job
                    job = Job(
                        job_id=generator.next_job_id,
                        complexity=complexity,
                        submit_time=current_time,
                        deadline=current_time + config.deadline_duration,
                        priority=priority
                    )

                    jobs.append(job)
                    generator.jobs.append(job)
                    generator.next_job_id += 1

                # Quiet period between bursts
                current_time += np.random.exponential(600)  # ~10 min quiet

            return jobs

        generator.generate = bursty_generate
        return generator

    @staticmethod
    def custom_scenario(
        num_jobs: int,
        target_utilization: float,
        total_gpus: int,
        avg_execution_time: float = AVG_EXECUTION_TIME,
        seed: Optional[int] = None
    ) -> JobGenerator:
        """
        Generate custom scenario with specified target utilization.

        Args:
            num_jobs: Number of jobs to generate
            target_utilization: Target utilization (0.0 to 1.0+)
            total_gpus: Total number of GPUs in fleet
            avg_execution_time: Average job execution time
            seed: Random seed for reproducibility

        Returns:
            Configured job generator
        """
        arrival_rate = WorkloadScenario.calculate_arrival_rate_for_utilization(
            target_utilization,
            total_gpus,
            avg_execution_time
        )

        config = JobGeneratorConfig(
            num_jobs=num_jobs,
            arrival_rate=arrival_rate,
            deadline_duration=DEFAULT_DEADLINE_DURATION,
            seed=seed
        )

        return JobGenerator(config)


def get_scenario_config(scenario_type: ScenarioType) -> ScenarioConfig:
    """
    Get predefined configuration for a scenario type.

    Args:
        scenario_type: Type of scenario

    Returns:
        Scenario configuration

    Raises:
        ValueError: If scenario type is not recognized
    """
    scenarios = {
        ScenarioType.LIGHT_LOAD: LIGHT_LOAD_CONFIG,
        ScenarioType.MODERATE_LOAD: MODERATE_LOAD_CONFIG,
        ScenarioType.HEAVY_LOAD: HEAVY_LOAD_CONFIG,
        ScenarioType.PEAK_HOUR: PEAK_HOUR_CONFIG,
    }

    if scenario_type not in scenarios:
        raise ValueError(
            f"Unknown scenario type: {scenario_type}. "
            f"Must be one of {list(scenarios.keys())}"
        )

    return scenarios[scenario_type]


if __name__ == "__main__":
    # Example usage
    print("CGRSP Workload Scenarios")
    print("=" * 50)

    # Test each scenario
    scenarios_to_test = [
        ("Light Load", lambda: WorkloadScenario.light_load(num_jobs=50, seed=42)),
        ("Moderate Load", lambda: WorkloadScenario.moderate_load(num_jobs=100, seed=42)),
        ("Heavy Load", lambda: WorkloadScenario.heavy_load(num_jobs=200, seed=42)),
        ("Peak Hour", lambda: WorkloadScenario.peak_hour(num_jobs=150, seed=42)),
    ]

    for name, scenario_fn in scenarios_to_test:
        print(f"\n{name}:")
        print("-" * 40)

        generator = scenario_fn()
        jobs = generator.generate()
        stats = generator.get_statistics()

        print(f"  Jobs: {stats['total_jobs']}")
        print(f"  Duration: {stats['duration_hours']:.2f} hours")
        print(f"  Complexity: {stats['complexity_distribution']}")
        print(f"  Avg inter-arrival: {stats['avg_inter_arrival_sec']:.2f}s")

        # Estimate utilization for 24 GPU fleet
        est_util = WorkloadScenario.estimate_utilization(
            generator.config.arrival_rate,
            num_gpus=24
        )
        print(f"  Est. utilization (24 GPUs): {est_util*100:.1f}%")
