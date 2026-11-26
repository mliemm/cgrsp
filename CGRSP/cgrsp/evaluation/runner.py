"""
Experiment Runner Module

Runs multiple simulation episodes with different schedulers,
collects results, and computes statistics.
"""

import numpy as np
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

from cgrsp.simulator.core import Simulator, EventType
from cgrsp.config import ResourceType, TemporalParameters, AvailabilityParameters
from cgrsp.data.execution_times import ExecutionTimeModel, GPUType
from cgrsp.evaluation.metrics import (
    PerformanceMetrics,
    QoSMetrics,
    OperationalMetrics,
    ObjectiveFunctionDecomposition,
    StatisticalAnalysis,
)


@dataclass
class ExperimentConfig:
    """
    Configuration for running experiments.

    Attributes:
        fleet_config: List of (ResourceType, count) tuples
        num_episodes: Number of simulation runs per scheduler
        simulation_duration: Length of each simulation (seconds)
        temporal_params: Temporal parameters
        availability_params: Availability parameters
        alpha_weight: QoS weight for objective function
        random_seeds: List of random seeds (one per episode)
        warmup_period: Warmup period excluded from metrics
        parallel_workers: Number of parallel workers (0 = sequential)
    """
    fleet_config: List[tuple] = field(default_factory=list)
    num_episodes: int = 30
    simulation_duration: float = 86400.0  # 24 hours
    temporal_params: Optional[TemporalParameters] = None
    availability_params: Optional[AvailabilityParameters] = None
    alpha_weight: float = 0.5
    random_seeds: Optional[List[int]] = None
    warmup_period: float = 0.0
    parallel_workers: int = 0  # 0 means sequential

    def __post_init__(self):
        """Generate random seeds if not provided."""
        if self.random_seeds is None:
            np.random.seed(42)
            self.random_seeds = [
                int(np.random.randint(0, 2**31 - 1))
                for _ in range(self.num_episodes)
            ]


@dataclass
class EpisodeResult:
    """
    Results from a single simulation episode.

    Attributes:
        episode_id: Episode number
        scheduler_name: Name of scheduler used
        random_seed: Random seed used
        qos_metrics: QoS metrics for this episode
        operational_metrics: Operational metrics for this episode
        objective_decomposition: Objective function decomposition
        execution_time: Wall-clock time to run episode (seconds)
    """
    episode_id: int
    scheduler_name: str
    random_seed: int
    qos_metrics: QoSMetrics
    operational_metrics: OperationalMetrics
    objective_decomposition: ObjectiveFunctionDecomposition
    execution_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "scheduler_name": self.scheduler_name,
            "random_seed": self.random_seed,
            "qos_metrics": self.qos_metrics.to_dict(),
            "operational_metrics": self.operational_metrics.to_dict(),
            "objective_decomposition": self.objective_decomposition.to_dict(),
            "execution_time": self.execution_time,
        }


@dataclass
class ExperimentResult:
    """
    Aggregated results from running multiple episodes.

    Attributes:
        scheduler_name: Name of scheduler
        config: Experiment configuration used
        episodes: List of individual episode results
        qos_mean: Mean QoS metrics across episodes
        operational_mean: Mean operational metrics across episodes
        objective_mean: Mean objective decomposition across episodes
        statistical_analysis: Statistical analysis results
        total_execution_time: Total wall-clock time
    """
    scheduler_name: str
    config: ExperimentConfig
    episodes: List[EpisodeResult] = field(default_factory=list)
    qos_mean: Optional[QoSMetrics] = None
    operational_mean: Optional[OperationalMetrics] = None
    objective_mean: Optional[ObjectiveFunctionDecomposition] = None
    statistical_analysis: Optional[StatisticalAnalysis] = None
    total_execution_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "scheduler_name": self.scheduler_name,
            "config": {
                "num_episodes": self.config.num_episodes,
                "simulation_duration": self.config.simulation_duration,
                "alpha_weight": self.config.alpha_weight,
            },
            "episodes": [ep.to_dict() for ep in self.episodes],
            "qos_mean": self.qos_mean.to_dict() if self.qos_mean else None,
            "operational_mean": self.operational_mean.to_dict() if self.operational_mean else None,
            "objective_mean": self.objective_mean.to_dict() if self.objective_mean else None,
            "statistical_analysis": self.statistical_analysis.to_dict() if self.statistical_analysis else None,
            "total_execution_time": self.total_execution_time,
        }


def _run_single_episode(
    scheduler_func: Callable,
    scheduler_name: str,
    config: ExperimentConfig,
    episode_id: int,
    random_seed: int,
    job_generator: Optional[Callable] = None,
) -> EpisodeResult:
    """
    Run a single simulation episode (for parallel execution).

    Args:
        scheduler_func: Scheduler function
        scheduler_name: Name of scheduler
        config: Experiment configuration
        episode_id: Episode number
        random_seed: Random seed for this episode
        job_generator: Optional job generator function

    Returns:
        EpisodeResult with metrics
    """
    start_time = time.time()

    # Create simulator
    simulator = Simulator(
        fleet_config=config.fleet_config,
        temporal_params=config.temporal_params,
        availability_params=config.availability_params,
        random_seed=random_seed,
    )

    # Set scheduler
    simulator.set_scheduler(scheduler_func)

    # Generate workload if job generator provided
    if job_generator:
        job_generator(simulator, config.simulation_duration)
    else:
        # Default simple workload
        _generate_default_workload(simulator, config.simulation_duration)

    # Run simulation
    simulator.run(config.simulation_duration, warmup_period=config.warmup_period)

    # Compute metrics
    metrics_calculator = PerformanceMetrics(alpha_weight=config.alpha_weight)
    qos = metrics_calculator.compute_qos_metrics(simulator.metrics_collector)

    # Get fleet size for operational metrics
    fleet_size = sum(count for _, count in config.fleet_config)
    operational = metrics_calculator.compute_operational_metrics(
        simulator.metrics_collector,
        fleet_size
    )

    objective = metrics_calculator.compute_objective_decomposition(qos, operational)

    execution_time = time.time() - start_time

    return EpisodeResult(
        episode_id=episode_id,
        scheduler_name=scheduler_name,
        random_seed=random_seed,
        qos_metrics=qos,
        operational_metrics=operational,
        objective_decomposition=objective,
        execution_time=execution_time,
    )


def _generate_default_workload(simulator: Simulator, duration: float) -> None:
    """
    Generate a default workload for testing.

    Args:
        simulator: Simulator instance
        duration: Simulation duration
    """
    # Submit 100 jobs throughout the simulation
    num_jobs = 100
    for i in range(num_jobs):
        arrival_time = (i / num_jobs) * duration

        # Random job parameters
        complexity = int(np.random.choice([1, 2, 3]))
        estimated_duration = ExecutionTimeModel.get_execution_time(complexity, GPUType.RTX_A5000)
        deadline = arrival_time + 28800.0   # 8 hour loose deadline

        simulator.schedule_event(
            EventType.JOB_ARRIVAL,
            arrival_time,
            {
                "complexity": complexity,
                "estimated_duration": estimated_duration,
                "deadline": deadline,
            },
        )


class ExperimentRunner:
    """
    Run experiments with multiple schedulers and collect results.

    Supports parallel execution for faster experiments.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: Dict[str, ExperimentResult] = {}

    def run_scheduler(
        self,
        scheduler_func: Callable,
        scheduler_name: str,
        job_generator: Optional[Callable] = None,
    ) -> ExperimentResult:
        """
        Run experiments for a single scheduler.

        Args:
            scheduler_func: Scheduler function
            scheduler_name: Name for this scheduler
            job_generator: Optional job generator

        Returns:
            ExperimentResult with aggregated results
        """
        start_time = time.time()
        episodes = []

        # Parallel or sequential execution
        if self.config.parallel_workers > 0:
            episodes = self._run_parallel(scheduler_func, scheduler_name, job_generator)
        else:
            episodes = self._run_sequential(scheduler_func, scheduler_name, job_generator)

        total_time = time.time() - start_time

        # Compute aggregate statistics
        metrics_calculator = PerformanceMetrics(alpha_weight=self.config.alpha_weight)

        qos_mean = self._compute_mean_qos(episodes)
        operational_mean = self._compute_mean_operational(episodes)
        objective_mean = self._compute_mean_objective(episodes)

        statistical = metrics_calculator.analyze_multiple_runs(
            [ep.qos_metrics for ep in episodes],
            [ep.operational_metrics for ep in episodes],
        )

        result = ExperimentResult(
            scheduler_name=scheduler_name,
            config=self.config,
            episodes=episodes,
            qos_mean=qos_mean,
            operational_mean=operational_mean,
            objective_mean=objective_mean,
            statistical_analysis=statistical,
            total_execution_time=total_time,
        )

        self.results[scheduler_name] = result
        return result

    def _run_sequential(
        self,
        scheduler_func: Callable,
        scheduler_name: str,
        job_generator: Optional[Callable] = None,
    ) -> List[EpisodeResult]:
        """Run episodes sequentially."""
        episodes = []

        for i in range(self.config.num_episodes):
            seed = self.config.random_seeds[i] if i < len(self.config.random_seeds) else i

            episode = _run_single_episode(
                scheduler_func=scheduler_func,
                scheduler_name=scheduler_name,
                config=self.config,
                episode_id=i,
                random_seed=seed,
                job_generator=job_generator,
            )

            episodes.append(episode)

        return episodes

    def _run_parallel(
        self,
        scheduler_func: Callable,
        scheduler_name: str,
        job_generator: Optional[Callable] = None,
    ) -> List[EpisodeResult]:
        """Run episodes in parallel using process pool."""
        episodes = [None] * self.config.num_episodes

        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all episodes
            futures = {}
            for i in range(self.config.num_episodes):
                seed = self.config.random_seeds[i] if i < len(self.config.random_seeds) else i

                future = executor.submit(
                    _run_single_episode,
                    scheduler_func,
                    scheduler_name,
                    self.config,
                    i,
                    seed,
                    job_generator,
                )
                futures[future] = i

            # Collect results as they complete
            for future in as_completed(futures):
                episode_id = futures[future]
                try:
                    episode = future.result()
                    episodes[episode_id] = episode
                except Exception as e:
                    print(f"Episode {episode_id} failed: {e}")

        # Filter out None values (failed episodes)
        return [ep for ep in episodes if ep is not None]

    def run_comparison(
        self,
        schedulers: Dict[str, Callable],
        job_generator: Optional[Callable] = None,
    ) -> Dict[str, ExperimentResult]:
        """
        Run experiments for multiple schedulers.

        Args:
            schedulers: Dictionary of scheduler names to functions
            job_generator: Optional job generator (same for all schedulers)

        Returns:
            Dictionary of scheduler names to results
        """
        print(f"Running comparison experiment with {len(schedulers)} schedulers...")
        print(f"Episodes per scheduler: {self.config.num_episodes}")
        print(f"Simulation duration: {self.config.simulation_duration}s")

        for name, func in schedulers.items():
            print(f"\nRunning scheduler: {name}...")
            self.run_scheduler(func, name, job_generator)
            print(f"Completed {name} in {self.results[name].total_execution_time:.2f}s")

        return self.results

    def compare_results(self) -> Dict[str, Any]:
        """
        Compare results across all schedulers.

        Returns:
            Dictionary with comparison metrics
        """
        if len(self.results) < 2:
            return {}

        metrics_calculator = PerformanceMetrics(alpha_weight=self.config.alpha_weight)

        comparison = {}

        # Get scheduler names
        names = list(self.results.keys())

        # Compare each pair
        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                result_a = self.results[name_a]
                result_b = self.results[name_b]

                # Paired tests
                paired_tests = metrics_calculator.compare_schedulers(
                    [ep.qos_metrics for ep in result_a.episodes],
                    [ep.qos_metrics for ep in result_b.episodes],
                    [ep.operational_metrics for ep in result_a.episodes],
                    [ep.operational_metrics for ep in result_b.episodes],
                )

                comparison[f"{name_a}_vs_{name_b}"] = {
                    "paired_tests": {
                        k: v.to_dict() for k, v in paired_tests.items()
                    }
                }

        return comparison

    def get_best_scheduler(self, metric: str = "objective_value") -> tuple[str, ExperimentResult]:
        """
        Find the best scheduler based on a metric.

        Args:
            metric: Metric to compare (objective_value, avg_response_time, total_cost, etc.)

        Returns:
            Tuple of (scheduler_name, result)
        """
        best_name = None
        best_value = None

        for name, result in self.results.items():
            if result.objective_mean is None:
                continue

            value = getattr(result.objective_mean, metric, None)
            if value is None:
                # Try QoS metrics
                if result.qos_mean:
                    value = getattr(result.qos_mean, metric, None)
                # Try operational metrics
                if value is None and result.operational_mean:
                    value = getattr(result.operational_mean, metric, None)

            if value is None:
                continue

            # Lower is better for objective value, response time, cost
            # Higher is better for adherence rate, utilization
            if metric in ["deadline_adherence_rate", "resource_utilization", "cost_efficiency"]:
                is_better = best_value is None or value > best_value
            else:
                is_better = best_value is None or value < best_value

            if is_better:
                best_value = value
                best_name = name

        return best_name, self.results[best_name] if best_name else None

    def _compute_mean_qos(self, episodes: List[EpisodeResult]) -> QoSMetrics:
        """Compute mean QoS metrics across episodes."""
        if not episodes:
            return QoSMetrics()

        return QoSMetrics(
            avg_waiting_time=np.mean([ep.qos_metrics.avg_waiting_time for ep in episodes]),
            avg_response_time=np.mean([ep.qos_metrics.avg_response_time for ep in episodes]),
            deadline_adherence_rate=np.mean([ep.qos_metrics.deadline_adherence_rate for ep in episodes]),
            p50_response_time=np.mean([ep.qos_metrics.p50_response_time for ep in episodes]),
            p95_response_time=np.mean([ep.qos_metrics.p95_response_time for ep in episodes]),
            p99_response_time=np.mean([ep.qos_metrics.p99_response_time for ep in episodes]),
            p50_waiting_time=np.mean([ep.qos_metrics.p50_waiting_time for ep in episodes]),
            p95_waiting_time=np.mean([ep.qos_metrics.p95_waiting_time for ep in episodes]),
            p99_waiting_time=np.mean([ep.qos_metrics.p99_waiting_time for ep in episodes]),
        )

    def _compute_mean_operational(self, episodes: List[EpisodeResult]) -> OperationalMetrics:
        """Compute mean operational metrics across episodes."""
        if not episodes:
            return OperationalMetrics()

        return OperationalMetrics(
            total_cost=np.mean([ep.operational_metrics.total_cost for ep in episodes]),
            avg_cost_per_job=np.mean([ep.operational_metrics.avg_cost_per_job for ep in episodes]),
            resource_utilization=np.mean([ep.operational_metrics.resource_utilization for ep in episodes]),
            cost_efficiency=np.mean([ep.operational_metrics.cost_efficiency for ep in episodes]),
            total_jobs_completed=int(np.mean([ep.operational_metrics.total_jobs_completed for ep in episodes])),
            total_jobs_submitted=int(np.mean([ep.operational_metrics.total_jobs_submitted for ep in episodes])),
            completion_rate=np.mean([ep.operational_metrics.completion_rate for ep in episodes]),
        )

    def _compute_mean_objective(self, episodes: List[EpisodeResult]) -> ObjectiveFunctionDecomposition:
        """Compute mean objective decomposition across episodes."""
        if not episodes:
            return ObjectiveFunctionDecomposition()

        return ObjectiveFunctionDecomposition(
            objective_value=np.mean([ep.objective_decomposition.objective_value for ep in episodes]),
            qos_component=np.mean([ep.objective_decomposition.qos_component for ep in episodes]),
            cost_component=np.mean([ep.objective_decomposition.cost_component for ep in episodes]),
            deadline_penalty=np.mean([ep.objective_decomposition.deadline_penalty for ep in episodes]),
            waiting_penalty=np.mean([ep.objective_decomposition.waiting_penalty for ep in episodes]),
            alpha_weight=episodes[0].objective_decomposition.alpha_weight,
        )

    def export_results(self, filepath: str) -> None:
        """
        Export results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json

        output_data = {
            "config": {
                "num_episodes": self.config.num_episodes,
                "simulation_duration": self.config.simulation_duration,
                "alpha_weight": self.config.alpha_weight,
            },
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary of results to console."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)

        for name, result in self.results.items():
            print(f"\nScheduler: {name}")
            print("-" * 40)
            print(f"Execution time: {result.total_execution_time:.2f}s")
            print(f"Episodes: {len(result.episodes)}")

            if result.qos_mean:
                print(f"\nQoS Metrics:")
                print(f"  Avg waiting time: {result.qos_mean.avg_waiting_time:.2f}s")
                print(f"  Avg response time: {result.qos_mean.avg_response_time:.2f}s")
                print(f"  Deadline adherence: {result.qos_mean.deadline_adherence_rate*100:.1f}%")
                print(f"  P95 response time: {result.qos_mean.p95_response_time:.2f}s")

            if result.operational_mean:
                print(f"\nOperational Metrics:")
                print(f"  Total cost: ${result.operational_mean.total_cost:.2f}")
                print(f"  Avg cost/job: ${result.operational_mean.avg_cost_per_job:.2f}")
                print(f"  Utilization: {result.operational_mean.resource_utilization*100:.1f}%")
                print(f"  Cost efficiency: {result.operational_mean.cost_efficiency:.2f} jobs/$")

            if result.objective_mean:
                print(f"\nObjective Function:")
                print(f"  Total Z: {result.objective_mean.objective_value:.4f}")
                print(f"  QoS component: {result.objective_mean.qos_component:.4f}")
                print(f"  Cost component: {result.objective_mean.cost_component:.4f}")

        print("\n" + "="*80)
