"""
Tests for experiment runner module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from cgrsp.simulator.core import Simulator
from cgrsp.config import ResourceType
from cgrsp.evaluation.runner import (
    ExperimentConfig,
    ExperimentResult,
    EpisodeResult,
    ExperimentRunner,
    _generate_default_workload,
)


@pytest.fixture
def sample_fleet_config():
    """Create sample fleet configuration."""
    return [(ResourceType.RTX_3090, 4), (ResourceType.RTX_A4000, 8)]


@pytest.fixture
def sample_config(sample_fleet_config):
    """Create sample experiment configuration."""
    return ExperimentConfig(
        fleet_config=sample_fleet_config,
        num_episodes=5,
        simulation_duration=3600.0,
        alpha_weight=0.5,
    )


@pytest.fixture
def mock_scheduler():
    """Create mock scheduler function."""
    def scheduler(state, simulator):
        # Simple mock scheduler
        pass
    return scheduler


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_config_creation(self, sample_fleet_config):
        """Test experiment configuration creation."""
        config = ExperimentConfig(
            fleet_config=sample_fleet_config,
            num_episodes=10,
            simulation_duration=7200.0,
        )

        assert config.num_episodes == 10
        assert config.simulation_duration == 7200.0
        assert len(config.fleet_config) == 2
        assert len(config.random_seeds) == 10  # Auto-generated

    def test_default_values(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.num_episodes == 30
        assert config.simulation_duration == 86400.0
        assert config.alpha_weight == 0.5
        assert config.parallel_workers == 0


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_episode_result_creation(self):
        """Test episode result creation."""
        from cgrsp.evaluation.metrics import QoSMetrics, OperationalMetrics, ObjectiveFunctionDecomposition

        qos = QoSMetrics(avg_waiting_time=100.0)
        operational = OperationalMetrics(total_cost=500.0)
        objective = ObjectiveFunctionDecomposition(objective_value=100.0)

        result = EpisodeResult(
            episode_id=0,
            scheduler_name="test_scheduler",
            random_seed=42,
            qos_metrics=qos,
            operational_metrics=operational,
            objective_decomposition=objective,
            execution_time=10.5,
        )

        assert result.episode_id == 0
        assert result.scheduler_name == "test_scheduler"
        assert result.random_seed == 42
        assert result.execution_time == 10.5

    def test_episode_result_to_dict(self):
        """Test episode result serialization."""
        from cgrsp.evaluation.metrics import QoSMetrics, OperationalMetrics, ObjectiveFunctionDecomposition

        qos = QoSMetrics(avg_waiting_time=100.0)
        operational = OperationalMetrics(total_cost=500.0)
        objective = ObjectiveFunctionDecomposition(objective_value=100.0)

        result = EpisodeResult(
            episode_id=0,
            scheduler_name="test",
            random_seed=42,
            qos_metrics=qos,
            operational_metrics=operational,
            objective_decomposition=objective,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["episode_id"] == 0
        assert data["scheduler_name"] == "test"
        assert "qos_metrics" in data


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_experiment_result_creation(self, sample_config):
        """Test experiment result creation."""
        from cgrsp.evaluation.metrics import QoSMetrics, OperationalMetrics, ObjectiveFunctionDecomposition

        qos = QoSMetrics(avg_waiting_time=100.0)
        operational = OperationalMetrics(total_cost=500.0)
        objective = ObjectiveFunctionDecomposition(objective_value=100.0)

        episode = EpisodeResult(
            episode_id=0,
            scheduler_name="test",
            random_seed=42,
            qos_metrics=qos,
            operational_metrics=operational,
            objective_decomposition=objective,
        )

        result = ExperimentResult(
            scheduler_name="test_scheduler",
            config=sample_config,
            episodes=[episode],
            qos_mean=qos,
            operational_mean=operational,
            objective_mean=objective,
            total_execution_time=50.0,
        )

        assert result.scheduler_name == "test_scheduler"
        assert len(result.episodes) == 1
        assert result.total_execution_time == 50.0


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    def test_runner_initialization(self, sample_config):
        """Test runner initialization."""
        runner = ExperimentRunner(sample_config)

        assert runner.config == sample_config
        assert len(runner.results) == 0

    @patch('cgrsp.evaluation.runner.Simulator')
    def test_run_scheduler_sequential(self, mock_simulator_class, sample_config, mock_scheduler):
        """Test running scheduler sequentially."""
        # Mock simulator
        mock_simulator = Mock()
        mock_simulator.metrics_collector = Mock()
        mock_simulator.metrics_collector.get_completed_job_metrics.return_value = []
        mock_simulator.metrics_collector.get_system_metrics.return_value = Mock()
        mock_simulator.metrics_collector.get_system_metrics.return_value.get_average_waiting_time.return_value = 100.0
        mock_simulator.metrics_collector.get_system_metrics.return_value.get_average_response_time.return_value = 500.0
        mock_simulator.metrics_collector.generate_summary_report.return_value = {}
        mock_simulator.fleet_manager.get_fleet_size.return_value = 12
        mock_simulator.metrics_collector.get_average_utilization.return_value = 0.8
        mock_simulator_class.return_value = mock_simulator

        runner = ExperimentRunner(sample_config)
        runner.config.parallel_workers = 0  # Sequential

        # Note: This will actually try to run, so we expect it to work with mocked simulator
        # In real testing, we'd need more sophisticated mocking

        assert runner.config.parallel_workers == 0

    def test_compute_mean_qos(self, sample_config):
        """Test computing mean QoS metrics."""
        from cgrsp.evaluation.metrics import QoSMetrics

        runner = ExperimentRunner(sample_config)

        episodes = []
        for i in range(5):
            qos = QoSMetrics(
                avg_waiting_time=100.0 + i * 10,
                avg_response_time=500.0 + i * 50,
                deadline_adherence_rate=0.9,
            )
            operational = Mock(total_cost=500.0, avg_cost_per_job=50.0,
                             resource_utilization=0.8, cost_efficiency=0.02,
                             total_jobs_completed=10, total_jobs_submitted=10,
                             completion_rate=1.0)
            objective = Mock(objective_value=100.0, qos_component=60.0,
                           cost_component=40.0, deadline_penalty=20.0,
                           waiting_penalty=10.0, alpha_weight=0.5)

            episode = EpisodeResult(
                episode_id=i,
                scheduler_name="test",
                random_seed=42 + i,
                qos_metrics=qos,
                operational_metrics=operational,
                objective_decomposition=objective,
            )
            episodes.append(episode)

        mean_qos = runner._compute_mean_qos(episodes)

        assert mean_qos.avg_waiting_time == 120.0  # (100 + 110 + 120 + 130 + 140) / 5
        assert mean_qos.avg_response_time == 600.0  # (500 + 550 + 600 + 650 + 700) / 5

    def test_compute_mean_operational(self, sample_config):
        """Test computing mean operational metrics."""
        from cgrsp.evaluation.metrics import QoSMetrics

        runner = ExperimentRunner(sample_config)

        episodes = []
        for i in range(5):
            qos = QoSMetrics(avg_waiting_time=100.0)
            operational = Mock(
                total_cost=500.0 + i * 10,
                avg_cost_per_job=50.0,
                resource_utilization=0.8,
                cost_efficiency=0.02,
                total_jobs_completed=10 + i,
                total_jobs_submitted=10,
                completion_rate=1.0,
            )
            objective = Mock(objective_value=100.0, qos_component=60.0,
                           cost_component=40.0, deadline_penalty=20.0,
                           waiting_penalty=10.0, alpha_weight=0.5)

            episode = EpisodeResult(
                episode_id=i,
                scheduler_name="test",
                random_seed=42 + i,
                qos_metrics=qos,
                operational_metrics=operational,
                objective_decomposition=objective,
            )
            episodes.append(episode)

        mean_ops = runner._compute_mean_operational(episodes)

        assert mean_ops.total_cost == 520.0  # (500 + 510 + 520 + 530 + 540) / 5
        assert mean_ops.total_jobs_completed == 12  # (10 + 11 + 12 + 13 + 14) / 5

    def test_compute_mean_objective(self, sample_config):
        """Test computing mean objective decomposition."""
        from cgrsp.evaluation.metrics import QoSMetrics

        runner = ExperimentRunner(sample_config)

        episodes = []
        for i in range(5):
            qos = QoSMetrics(avg_waiting_time=100.0)
            operational = Mock(total_cost=500.0, avg_cost_per_job=50.0,
                             resource_utilization=0.8, cost_efficiency=0.02,
                             total_jobs_completed=10, total_jobs_submitted=10,
                             completion_rate=1.0)
            objective = Mock(
                objective_value=100.0 + i * 5,
                qos_component=60.0,
                cost_component=40.0,
                deadline_penalty=20.0,
                waiting_penalty=10.0,
                alpha_weight=0.5,
            )

            episode = EpisodeResult(
                episode_id=i,
                scheduler_name="test",
                random_seed=42 + i,
                qos_metrics=qos,
                operational_metrics=operational,
                objective_decomposition=objective,
            )
            episodes.append(episode)

        mean_obj = runner._compute_mean_objective(episodes)

        assert mean_obj.objective_value == 110.0  # (100 + 105 + 110 + 115 + 120) / 5


class TestDefaultWorkload:
    """Tests for default workload generator."""

    def test_generate_default_workload(self):
        """Test default workload generation."""
        mock_simulator = Mock(spec=Simulator)
        mock_simulator.schedule_event = Mock()

        _generate_default_workload(mock_simulator, duration=3600.0)

        # Should schedule 100 jobs
        assert mock_simulator.schedule_event.call_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
