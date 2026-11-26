"""
Tests for evaluation metrics module.
"""

import pytest
import numpy as np
from cgrsp.simulator.metrics import MetricsCollector, JobMetrics, SystemMetrics
from cgrsp.evaluation.metrics import (
    PerformanceMetrics,
    QoSMetrics,
    OperationalMetrics,
    ObjectiveFunctionDecomposition,
    StatisticalAnalysis,
    ConfidenceInterval,
    StatisticalTestResult,
)


@pytest.fixture
def sample_job_metrics():
    """Create sample job metrics for testing."""
    jobs = []
    for i in range(10):
        job = JobMetrics(
            job_id=i,
            submit_time=i * 1000.0,
            deadline=(i * 1000.0) + 5000.0,
            start_time=(i * 1000.0) + 500.0,
            completion_time=(i * 1000.0) + 2000.0,
            execution_cost=10.0 + i,
        )
        job.calculate_metrics()
        jobs.append(job)
    return jobs


@pytest.fixture
def sample_metrics_collector(sample_job_metrics):
    """Create a sample metrics collector."""
    collector = MetricsCollector()

    for job in sample_job_metrics:
        collector._job_metrics[job.job_id] = job
        collector.system_metrics.total_jobs_completed += 1
        collector.system_metrics.total_jobs_submitted += 1
        collector.system_metrics.total_execution_cost += job.execution_cost
        collector.system_metrics.waiting_times.append(job.waiting_time)
        collector.system_metrics.response_times.append(job.response_time)

    return collector


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_initialization(self):
        """Test metrics calculator initialization."""
        calc = PerformanceMetrics(alpha_weight=0.7)
        assert calc.alpha_weight == 0.7
        assert calc.max_waiting_threshold == 3600.0

    def test_compute_qos_metrics(self, sample_metrics_collector):
        """Test QoS metrics computation."""
        calc = PerformanceMetrics()
        qos = calc.compute_qos_metrics(sample_metrics_collector)

        assert isinstance(qos, QoSMetrics)
        assert qos.avg_waiting_time > 0
        assert qos.avg_response_time > 0
        assert qos.deadline_adherence_rate >= 0
        assert qos.deadline_adherence_rate <= 1
        assert len(qos.response_time_distribution) == 10
        assert len(qos.waiting_time_distribution) == 10

    def test_compute_operational_metrics(self, sample_metrics_collector):
        """Test operational metrics computation."""
        calc = PerformanceMetrics()
        operational = calc.compute_operational_metrics(sample_metrics_collector, fleet_size=20)

        assert isinstance(operational, OperationalMetrics)
        assert operational.total_cost > 0
        assert operational.avg_cost_per_job > 0
        assert operational.resource_utilization >= 0
        assert operational.total_jobs_completed == 10

    def test_compute_objective_decomposition(self):
        """Test objective function decomposition."""
        calc = PerformanceMetrics(alpha_weight=0.5)

        qos = QoSMetrics(
            avg_waiting_time=1000.0,
            avg_response_time=2000.0,
            deadline_adherence_rate=0.95,
        )

        operational = OperationalMetrics(
            total_cost=500.0,
            avg_cost_per_job=50.0,
        )

        objective = calc.compute_objective_decomposition(qos, operational)

        assert isinstance(objective, ObjectiveFunctionDecomposition)
        assert objective.objective_value > 0
        assert objective.qos_component >= 0
        assert objective.cost_component >= 0
        assert objective.alpha_weight == 0.5

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        calc = PerformanceMetrics()

        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        ci = calc.compute_confidence_interval(values, confidence_level=0.95)

        assert isinstance(ci, ConfidenceInterval)
        assert ci.mean > 0
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean
        assert ci.confidence_level == 0.95

    def test_confidence_interval_insufficient_data(self):
        """Test confidence interval with insufficient data."""
        calc = PerformanceMetrics()

        ci_empty = calc.compute_confidence_interval([])
        assert ci_empty.mean == 0.0

        ci_single = calc.compute_confidence_interval([10.0])
        assert ci_single.mean == 10.0
        assert ci_single.lower == 0.0
        assert ci_single.upper == 0.0

    def test_paired_t_test(self):
        """Test paired t-test computation."""
        calc = PerformanceMetrics()

        sample1 = [10.0, 12.0, 11.0, 13.0, 9.0]
        sample2 = [11.0, 13.0, 12.0, 14.0, 10.0]

        result = calc.compute_paired_t_test(sample1, sample2)

        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Paired t-test"
        assert result.p_value >= 0
        assert result.p_value <= 1
        assert isinstance(result.is_significant, bool)

    def test_paired_t_test_insufficient_data(self):
        """Test paired t-test with insufficient data."""
        calc = PerformanceMetrics()

        result = calc.compute_paired_t_test([1.0], [2.0])

        assert result.is_significant is False
        assert result.interpretation == "Insufficient data for test"

    def test_effect_size(self):
        """Test effect size computation."""
        calc = PerformanceMetrics()

        sample1 = [10.0, 12.0, 11.0, 13.0, 9.0]
        sample2 = [15.0, 17.0, 16.0, 18.0, 14.0]

        effect_size = calc.compute_effect_size(sample1, sample2)

        assert effect_size > 0
        assert effect_size < 10  # Reasonable upper bound

    def test_analyze_multiple_runs(self):
        """Test statistical analysis across multiple runs."""
        calc = PerformanceMetrics()

        qos_list = []
        operational_list = []

        for i in range(10):
            qos = QoSMetrics(
                avg_waiting_time=1000.0 + i * 10,
                avg_response_time=2000.0 + i * 20,
                deadline_adherence_rate=0.95 - i * 0.01,
            )
            operational = OperationalMetrics(
                total_cost=500.0 + i * 10,
                avg_cost_per_job=50.0,
            )
            qos_list.append(qos)
            operational_list.append(operational)

        analysis = calc.analyze_multiple_runs(qos_list, operational_list)

        assert isinstance(analysis, StatisticalAnalysis)
        assert len(analysis.confidence_intervals) > 0
        assert analysis.sample_size == 10

    def test_compare_schedulers(self):
        """Test scheduler comparison."""
        calc = PerformanceMetrics()

        # Create mock data for two schedulers
        scheduler_a_qos = [QoSMetrics(avg_waiting_time=1000.0, deadline_adherence_rate=0.9) for _ in range(10)]
        scheduler_b_qos = [QoSMetrics(avg_waiting_time=1200.0, deadline_adherence_rate=0.85) for _ in range(10)]
        scheduler_a_ops = [OperationalMetrics(total_cost=500.0, resource_utilization=0.8) for _ in range(10)]
        scheduler_b_ops = [OperationalMetrics(total_cost=550.0, resource_utilization=0.75) for _ in range(10)]

        comparison = calc.compare_schedulers(
            scheduler_a_qos, scheduler_b_qos,
            scheduler_a_ops, scheduler_b_ops
        )

        assert len(comparison) == 5  # Number of metrics compared
        for metric_name, result in comparison.items():
            assert isinstance(result, StatisticalTestResult)


class TestQoSMetrics:
    """Tests for QoSMetrics dataclass."""

    def test_qos_metrics_creation(self):
        """Test QoS metrics object creation."""
        qos = QoSMetrics(
            avg_waiting_time=100.0,
            avg_response_time=500.0,
            deadline_adherence_rate=0.95,
        )

        assert qos.avg_waiting_time == 100.0
        assert qos.avg_response_time == 500.0
        assert qos.deadline_adherence_rate == 0.95

    def test_qos_metrics_to_dict(self):
        """Test QoS metrics serialization."""
        qos = QoSMetrics(
            avg_waiting_time=100.0,
            avg_response_time=500.0,
            deadline_adherence_rate=0.95,
            p50_response_time=450.0,
            p95_response_time=800.0,
        )

        data = qos.to_dict()

        assert isinstance(data, dict)
        assert data["avg_waiting_time"] == 100.0
        assert data["avg_response_time"] == 500.0
        assert data["deadline_adherence_rate"] == 0.95


class TestOperationalMetrics:
    """Tests for OperationalMetrics dataclass."""

    def test_operational_metrics_creation(self):
        """Test operational metrics object creation."""
        ops = OperationalMetrics(
            total_cost=1000.0,
            avg_cost_per_job=50.0,
            resource_utilization=0.85,
        )

        assert ops.total_cost == 1000.0
        assert ops.avg_cost_per_job == 50.0
        assert ops.resource_utilization == 0.85

    def test_operational_metrics_to_dict(self):
        """Test operational metrics serialization."""
        ops = OperationalMetrics(
            total_cost=1000.0,
            avg_cost_per_job=50.0,
            resource_utilization=0.85,
            cost_efficiency=0.02,
        )

        data = ops.to_dict()

        assert isinstance(data, dict)
        assert data["total_cost"] == 1000.0
        assert data["resource_utilization"] == 0.85


class TestObjectiveFunctionDecomposition:
    """Tests for ObjectiveFunctionDecomposition dataclass."""

    def test_objective_decomposition_creation(self):
        """Test objective decomposition creation."""
        obj = ObjectiveFunctionDecomposition(
            objective_value=100.0,
            qos_component=60.0,
            cost_component=40.0,
            alpha_weight=0.6,
        )

        assert obj.objective_value == 100.0
        assert obj.qos_component == 60.0
        assert obj.cost_component == 40.0
        assert obj.alpha_weight == 0.6

    def test_objective_decomposition_to_dict(self):
        """Test objective decomposition serialization."""
        obj = ObjectiveFunctionDecomposition(
            objective_value=100.0,
            qos_component=60.0,
            cost_component=40.0,
            deadline_penalty=20.0,
            waiting_penalty=10.0,
        )

        data = obj.to_dict()

        assert isinstance(data, dict)
        assert data["objective_value"] == 100.0
        assert data["qos_component"] == 60.0


class TestMetricsCollectorTardiness:
    """Tests for aggregate tardiness fields in generate_summary_report."""

    def test_summary_reports_tardiness(self):
        """Two on-time jobs and one job 120 s late: avg=40.0, p95>=0.0."""
        collector = MetricsCollector()

        # Job 0: submit=0, deadline=1000, start=0, complete=900 (on time)
        collector.record_job_submission(job_id=0, submit_time=0.0, deadline=1000.0)
        collector.record_job_start(job_id=0, start_time=0.0, node_id=1, resource_type="A100")
        collector.record_job_completion(job_id=0, completion_time=900.0, execution_cost=1.0)

        # Job 1: submit=0, deadline=1000, start=0, complete=800 (on time)
        collector.record_job_submission(job_id=1, submit_time=0.0, deadline=1000.0)
        collector.record_job_start(job_id=1, start_time=0.0, node_id=1, resource_type="A100")
        collector.record_job_completion(job_id=1, completion_time=800.0, execution_cost=1.0)

        # Job 2: submit=0, deadline=1000, start=0, complete=1120 (120 s late)
        collector.record_job_submission(job_id=2, submit_time=0.0, deadline=1000.0)
        collector.record_job_start(job_id=2, start_time=0.0, node_id=1, resource_type="A100")
        collector.record_job_completion(job_id=2, completion_time=1120.0, execution_cost=1.0)

        summary = collector.generate_summary_report()
        assert summary["avg_tardiness_s"] == 40.0        # (0 + 0 + 120) / 3
        assert summary["p95_tardiness_s"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
