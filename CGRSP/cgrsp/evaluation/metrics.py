"""
Performance Metrics Module

Computes QoS metrics, operational metrics, and performs statistical analysis
for evaluating CGRSP scheduling algorithms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats
from cgrsp.simulator.metrics import MetricsCollector, JobMetrics


@dataclass
class QoSMetrics:
    """
    Quality of Service metrics for evaluating scheduling performance.

    Attributes:
        avg_waiting_time: Average time jobs spend waiting in queue
        avg_response_time: Average time from submission to completion
        deadline_adherence_rate: Percentage of jobs meeting deadlines
        response_time_distribution: Full distribution of response times
        waiting_time_distribution: Full distribution of waiting times
        p50_response_time: Median response time
        p95_response_time: 95th percentile response time
        p99_response_time: 99th percentile response time
        p50_waiting_time: Median waiting time
        p95_waiting_time: 95th percentile waiting time
        p99_waiting_time: 99th percentile waiting time
    """
    avg_waiting_time: float = 0.0
    avg_response_time: float = 0.0
    deadline_adherence_rate: float = 0.0
    response_time_distribution: List[float] = field(default_factory=list)
    waiting_time_distribution: List[float] = field(default_factory=list)
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    p50_waiting_time: float = 0.0
    p95_waiting_time: float = 0.0
    p99_waiting_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "avg_waiting_time": self.avg_waiting_time,
            "avg_response_time": self.avg_response_time,
            "deadline_adherence_rate": self.deadline_adherence_rate,
            "p50_response_time": self.p50_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "p50_waiting_time": self.p50_waiting_time,
            "p95_waiting_time": self.p95_waiting_time,
            "p99_waiting_time": self.p99_waiting_time,
        }


@dataclass
class OperationalMetrics:
    """
    Operational metrics for cost and resource efficiency.

    Attributes:
        total_cost: Total execution cost across all jobs
        avg_cost_per_job: Average cost per completed job
        resource_utilization: Average fleet utilization (0-1)
        cost_efficiency: Jobs completed per dollar
        total_jobs_completed: Number of completed jobs
        total_jobs_submitted: Number of submitted jobs
        completion_rate: Percentage of submitted jobs that completed
    """
    total_cost: float = 0.0
    avg_cost_per_job: float = 0.0
    resource_utilization: float = 0.0
    cost_efficiency: float = 0.0  # jobs per dollar
    total_jobs_completed: int = 0
    total_jobs_submitted: int = 0
    completion_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_cost": self.total_cost,
            "avg_cost_per_job": self.avg_cost_per_job,
            "resource_utilization": self.resource_utilization,
            "cost_efficiency": self.cost_efficiency,
            "total_jobs_completed": self.total_jobs_completed,
            "total_jobs_submitted": self.total_jobs_submitted,
            "completion_rate": self.completion_rate,
        }


@dataclass
class ObjectiveFunctionDecomposition:
    """
    Decomposition of the objective function into QoS and cost components.

    NOTE: This is a legacy normalised diagnostic from the earlier evaluation
    harness (Z = alpha * QoS_penalty + (1 - alpha) * cost with alpha = 0.5 and a
    normalised deadline penalty). It is NOT the model objective used for any
    thesis result. The thesis numbers come from the phase*.py scripts, which use
    cgrsp.simulator.metrics.MetricsCollector and the lexicographic QoS-primary
    objective of Eq. (objective) with alpha_1 = 1, alpha_2 = 10 (and a symmetric
    weighted-sum 0.5/0.5 robustness variant). This class is retained only for the
    older evaluation.runner / evaluation.sensitivity code path and its tests.

    The legacy objective function is: Z = α * QoS_penalty + (1-α) * cost
    where QoS_penalty includes deadline misses and waiting time.

    Attributes:
        objective_value: Total objective function value Z
        qos_component: QoS penalty component
        cost_component: Cost component
        deadline_penalty: Penalty from deadline misses
        waiting_penalty: Penalty from excessive waiting
        alpha_weight: QoS weight parameter (α)
    """
    objective_value: float = 0.0
    qos_component: float = 0.0
    cost_component: float = 0.0
    deadline_penalty: float = 0.0
    waiting_penalty: float = 0.0
    alpha_weight: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "objective_value": self.objective_value,
            "qos_component": self.qos_component,
            "cost_component": self.cost_component,
            "deadline_penalty": self.deadline_penalty,
            "waiting_penalty": self.waiting_penalty,
            "alpha_weight": self.alpha_weight,
        }


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    mean: float
    lower: float
    upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
        }


@dataclass
class StatisticalTestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "interpretation": self.interpretation,
        }


@dataclass
class StatisticalAnalysis:
    """
    Statistical analysis results for comparing schedulers.

    Attributes:
        confidence_intervals: 95% CI for key metrics
        paired_tests: Results of paired t-tests between schedulers
        effect_sizes: Cohen's d effect sizes
        sample_size: Number of samples (episodes)
    """
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    paired_tests: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence_intervals": {
                k: v.to_dict() for k, v in self.confidence_intervals.items()
            },
            "paired_tests": {
                k: v.to_dict() for k, v in self.paired_tests.items()
            },
            "effect_sizes": self.effect_sizes,
            "sample_size": self.sample_size,
        }


class PerformanceMetrics:
    """
    Compute performance metrics from simulation results.

    Handles QoS metrics, operational metrics, objective function
    decomposition, and statistical analysis.
    """

    def __init__(self, alpha_weight: float = 0.5, max_waiting_threshold: float = 3600.0):
        """
        Initialize metrics calculator.

        Args:
            alpha_weight: QoS weight α in [0, 1] for objective function
            max_waiting_threshold: Threshold for waiting time penalty (seconds)
        """
        self.alpha_weight = alpha_weight
        self.max_waiting_threshold = max_waiting_threshold

    def compute_qos_metrics(self, metrics_collector: MetricsCollector) -> QoSMetrics:
        """
        Compute QoS metrics from collected data.

        Args:
            metrics_collector: Metrics collector from simulation run

        Returns:
            QoSMetrics object with all QoS metrics
        """
        system_metrics = metrics_collector.get_system_metrics()
        job_metrics = metrics_collector.get_completed_job_metrics()

        if not job_metrics:
            return QoSMetrics()

        waiting_times = [job.waiting_time for job in job_metrics]
        response_times = [job.response_time for job in job_metrics]

        # Calculate adherence rate
        jobs_met_deadline = sum(1 for job in job_metrics if job.deadline_met)
        adherence_rate = jobs_met_deadline / len(job_metrics) if job_metrics else 0.0

        # Calculate percentiles
        p50_wait = np.percentile(waiting_times, 50) if waiting_times else 0.0
        p95_wait = np.percentile(waiting_times, 95) if waiting_times else 0.0
        p99_wait = np.percentile(waiting_times, 99) if waiting_times else 0.0

        p50_resp = np.percentile(response_times, 50) if response_times else 0.0
        p95_resp = np.percentile(response_times, 95) if response_times else 0.0
        p99_resp = np.percentile(response_times, 99) if response_times else 0.0

        return QoSMetrics(
            avg_waiting_time=system_metrics.get_average_waiting_time(),
            avg_response_time=system_metrics.get_average_response_time(),
            deadline_adherence_rate=adherence_rate,
            response_time_distribution=response_times,
            waiting_time_distribution=waiting_times,
            p50_response_time=p50_resp,
            p95_response_time=p95_resp,
            p99_response_time=p99_resp,
            p50_waiting_time=p50_wait,
            p95_waiting_time=p95_wait,
            p99_waiting_time=p99_wait,
        )

    def compute_operational_metrics(
        self,
        metrics_collector: MetricsCollector,
        fleet_size: int
    ) -> OperationalMetrics:
        """
        Compute operational metrics from collected data.

        Args:
            metrics_collector: Metrics collector from simulation run
            fleet_size: Total number of nodes in fleet

        Returns:
            OperationalMetrics object with all operational metrics
        """
        system_metrics = metrics_collector.get_system_metrics()
        summary = metrics_collector.generate_summary_report()

        total_cost = system_metrics.total_execution_cost
        completed_jobs = system_metrics.total_jobs_completed
        submitted_jobs = system_metrics.total_jobs_submitted

        avg_cost = total_cost / completed_jobs if completed_jobs > 0 else 0.0
        utilization = metrics_collector.get_average_utilization()
        cost_efficiency = completed_jobs / total_cost if total_cost > 0 else 0.0
        completion_rate = completed_jobs / submitted_jobs if submitted_jobs > 0 else 0.0

        return OperationalMetrics(
            total_cost=total_cost,
            avg_cost_per_job=avg_cost,
            resource_utilization=utilization,
            cost_efficiency=cost_efficiency,
            total_jobs_completed=completed_jobs,
            total_jobs_submitted=submitted_jobs,
            completion_rate=completion_rate,
        )

    def compute_objective_decomposition(
        self,
        qos_metrics: QoSMetrics,
        operational_metrics: OperationalMetrics,
        deadline_penalty_weight: float = 1000.0,
    ) -> ObjectiveFunctionDecomposition:
        """
        Decompose objective function into components.

        Legacy diagnostic only (see ObjectiveFunctionDecomposition): this is not
        the model's lexicographic objective and feeds no thesis result.

        Objective: Z = α * QoS_penalty + (1-α) * cost

        QoS_penalty = deadline_penalty_weight * (1 - deadline_adherence_rate)
                     + waiting_penalty_weight * avg_waiting_time / threshold

        Args:
            qos_metrics: Computed QoS metrics
            operational_metrics: Computed operational metrics
            deadline_penalty_weight: Weight for deadline misses

        Returns:
            ObjectiveFunctionDecomposition with all components
        """
        # Calculate deadline penalty (normalized)
        deadline_miss_rate = 1.0 - qos_metrics.deadline_adherence_rate
        deadline_penalty = deadline_penalty_weight * deadline_miss_rate

        # Calculate waiting penalty (normalized by threshold)
        waiting_penalty = qos_metrics.avg_waiting_time / self.max_waiting_threshold

        # QoS component combines both penalties
        qos_component = deadline_penalty + waiting_penalty

        # Cost component (normalized to reasonable scale)
        # Assume baseline cost of $100 for normalization
        cost_component = operational_metrics.total_cost / 100.0

        # Total objective
        objective_value = (
            self.alpha_weight * qos_component +
            (1 - self.alpha_weight) * cost_component
        )

        return ObjectiveFunctionDecomposition(
            objective_value=objective_value,
            qos_component=qos_component,
            cost_component=cost_component,
            deadline_penalty=deadline_penalty,
            waiting_penalty=waiting_penalty,
            alpha_weight=self.alpha_weight,
        )

    def compute_confidence_interval(
        self,
        values: List[float],
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for a set of values.

        Legacy helper, not used for any thesis result (those come from
        phase3_statistical.py). For a single value the band is degenerate
        (lower = upper = 0.0), not a valid interval.

        Args:
            values: List of metric values from multiple runs
            confidence_level: Confidence level (default: 0.95 for 95% CI)

        Returns:
            ConfidenceInterval with mean, lower, and upper bounds
        """
        if not values:
            return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0, confidence_level=confidence_level)
        if len(values) == 1:
            return ConfidenceInterval(mean=float(values[0]), lower=0.0, upper=0.0, confidence_level=confidence_level)

        values_array = np.array(values)
        mean = np.mean(values_array)
        std_err = stats.sem(values_array)

        # Use t-distribution for small samples
        alpha = 1 - confidence_level
        df = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)

        margin_of_error = t_critical * std_err

        return ConfidenceInterval(
            mean=float(mean),
            lower=float(mean - margin_of_error),
            upper=float(mean + margin_of_error),
            confidence_level=confidence_level,
        )

    def compute_paired_t_test(
        self,
        sample1: List[float],
        sample2: List[float],
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Perform paired t-test between two samples.

        Legacy helper, not used for any thesis result. The effect size here
        is mean(diff) / population SD(diff) (d_z, ddof=0), not the pooled-SD
        Cohen's d that the thesis reports.

        Args:
            sample1: First sample (e.g., scheduler A metrics)
            sample2: Second sample (e.g., scheduler B metrics)
            alpha: Significance level (default: 0.05)

        Returns:
            StatisticalTestResult with test statistics
        """
        if len(sample1) != len(sample2) or len(sample1) < 2:
            return StatisticalTestResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
                interpretation="Insufficient data for test",
            )

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(sample1, sample2)

        # Calculate Cohen's d for effect size
        diff_array = np.array(sample1) - np.array(sample2)
        effect_size = np.mean(diff_array) / np.std(diff_array) if np.std(diff_array) > 0 else 0.0

        # Interpret effect size
        if abs(effect_size) < 0.2:
            interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            interpretation = "small"
        elif abs(effect_size) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        is_significant = bool(p_value < alpha)

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(effect_size),
            interpretation=f"{interpretation} effect size",
        )

    def analyze_multiple_runs(
        self,
        qos_metrics_list: List[QoSMetrics],
        operational_metrics_list: List[OperationalMetrics],
    ) -> StatisticalAnalysis:
        """
        Perform statistical analysis across multiple runs.

        Args:
            qos_metrics_list: List of QoS metrics from multiple runs
            operational_metrics_list: List of operational metrics from multiple runs

        Returns:
            StatisticalAnalysis with confidence intervals and test results
        """
        sample_size = len(qos_metrics_list)

        # Extract metric values across runs
        waiting_times = [m.avg_waiting_time for m in qos_metrics_list]
        response_times = [m.avg_response_time for m in qos_metrics_list]
        adherence_rates = [m.deadline_adherence_rate for m in qos_metrics_list]
        total_costs = [m.total_cost for m in operational_metrics_list]
        utilizations = [m.resource_utilization for m in operational_metrics_list]

        # Compute confidence intervals
        confidence_intervals = {
            "avg_waiting_time": self.compute_confidence_interval(waiting_times),
            "avg_response_time": self.compute_confidence_interval(response_times),
            "deadline_adherence_rate": self.compute_confidence_interval(adherence_rates),
            "total_cost": self.compute_confidence_interval(total_costs),
            "resource_utilization": self.compute_confidence_interval(utilizations),
        }

        return StatisticalAnalysis(
            confidence_intervals=confidence_intervals,
            sample_size=sample_size,
        )

    def compare_schedulers(
        self,
        scheduler_a_qos: List[QoSMetrics],
        scheduler_b_qos: List[QoSMetrics],
        scheduler_a_ops: List[OperationalMetrics],
        scheduler_b_ops: List[OperationalMetrics],
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare two schedulers using paired statistical tests.

        Args:
            scheduler_a_qos: QoS metrics from scheduler A runs
            scheduler_b_qos: QoS metrics from scheduler B runs
            scheduler_a_ops: Operational metrics from scheduler A runs
            scheduler_b_ops: Operational metrics from scheduler B runs

        Returns:
            Dictionary of metric names to test results
        """
        # Extract metric values
        a_waiting = [m.avg_waiting_time for m in scheduler_a_qos]
        b_waiting = [m.avg_waiting_time for m in scheduler_b_qos]

        a_response = [m.avg_response_time for m in scheduler_a_qos]
        b_response = [m.avg_response_time for m in scheduler_b_qos]

        a_adherence = [m.deadline_adherence_rate for m in scheduler_a_qos]
        b_adherence = [m.deadline_adherence_rate for m in scheduler_b_qos]

        a_cost = [m.total_cost for m in scheduler_a_ops]
        b_cost = [m.total_cost for m in scheduler_b_ops]

        a_utilization = [m.resource_utilization for m in scheduler_a_ops]
        b_utilization = [m.resource_utilization for m in scheduler_b_ops]

        # Perform tests
        paired_tests = {
            "avg_waiting_time": self.compute_paired_t_test(a_waiting, b_waiting),
            "avg_response_time": self.compute_paired_t_test(a_response, b_response),
            "deadline_adherence_rate": self.compute_paired_t_test(a_adherence, b_adherence),
            "total_cost": self.compute_paired_t_test(a_cost, b_cost),
            "resource_utilization": self.compute_paired_t_test(a_utilization, b_utilization),
        }

        return paired_tests

    def compute_effect_size(
        self,
        sample1: List[float],
        sample2: List[float]
    ) -> float:
        """
        Compute the absolute pooled-SD Cohen's d between two samples.

        Legacy helper, not used for any thesis result; the sign is discarded.

        Args:
            sample1: First sample
            sample2: Second sample

        Returns:
            Cohen's d effect size
        """
        if not sample1 or not sample2:
            return 0.0

        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)

        # Pooled standard deviation
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return abs((mean1 - mean2) / pooled_std)
