"""
Tests for visualization module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from cgrsp.evaluation.visualization import (
    PerformanceVisualizer,
    TimeSeriesVisualizer,
    SensitivityVisualizer,
)


@pytest.fixture
def sample_results():
    """Create sample experiment results for visualization."""
    from cgrsp.evaluation.runner import ExperimentResult, ExperimentConfig, EpisodeResult
    from cgrsp.evaluation.metrics import QoSMetrics, OperationalMetrics, ObjectiveFunctionDecomposition
    from cgrsp.config import ResourceType

    config = ExperimentConfig(
        fleet_config=[(ResourceType.RTX_3090, 4)],
        num_episodes=5,
        simulation_duration=3600.0,
    )

    qos = QoSMetrics(
        avg_waiting_time=100.0,
        avg_response_time=500.0,
        deadline_adherence_rate=0.95,
        p95_response_time=800.0,
    )

    operational = OperationalMetrics(
        total_cost=500.0,
        avg_cost_per_job=50.0,
        resource_utilization=0.85,
    )

    objective = ObjectiveFunctionDecomposition(
        objective_value=100.0,
        qos_component=60.0,
        cost_component=40.0,
    )

    episode = EpisodeResult(
        episode_id=0,
        scheduler_name="fifo",
        random_seed=42,
        qos_metrics=qos,
        operational_metrics=operational,
        objective_decomposition=objective,
    )

    result = ExperimentResult(
        scheduler_name="fifo",
        config=config,
        episodes=[episode],
        qos_mean=qos,
        operational_mean=operational,
        objective_mean=objective,
    )

    return {"fifo": result}


@pytest.fixture
def sample_sensitivity_result():
    """Create sample sensitivity result."""
    from cgrsp.evaluation.sensitivity import ParameterSweep, SensitivityResult

    sweep1 = ParameterSweep(
        parameter_name="fleet_size",
        parameter_values=[0.5, 1.0, 1.5],
        objective_values=[120.0, 100.0, 80.0],
        qos_values=[1200.0, 1000.0, 800.0],
        cost_values=[600.0, 500.0, 400.0],
        sensitivity_scores=[0.5, 0.0, 0.5],
    )

    sweep2 = ParameterSweep(
        parameter_name="availability",
        parameter_values=[0.85, 0.9, 0.95],
        objective_values=[110.0, 100.0, 95.0],
        qos_values=[1100.0, 1000.0, 950.0],
        cost_values=[550.0, 500.0, 475.0],
        sensitivity_scores=[0.3, 0.0, 0.2],
    )

    result = SensitivityResult(
        parameter_sweeps={
            "fleet_size": sweep1,
            "availability": sweep2,
        },
        high_sensitivity_params=[],
        sensitivity_ranking=[
            ("fleet_size", 0.33),
            ("availability", 0.17),
        ],
        baseline_objective=100.0,
    )

    return result


class TestPerformanceVisualizer:
    """Tests for PerformanceVisualizer class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        viz = PerformanceVisualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_metric_comparison(self, mock_savefig, mock_show, sample_results):
        """Test plotting metric comparison."""
        viz = PerformanceVisualizer()

        # Should not raise exceptions
        viz.plot_metric_comparison(
            sample_results,
            metrics=["avg_response_time", "deadline_adherence_rate"],
            show=False,
        )

        # Verify show was not called (show=False)
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_box_comparison(self, mock_show):
        """Test plotting box comparison."""
        viz = PerformanceVisualizer()

        scheduler_data = {
            "fifo": [100.0, 110.0, 105.0, 98.0, 102.0],
            "priority": [90.0, 95.0, 92.0, 88.0, 91.0],
        }

        # Should not raise exceptions
        viz.plot_box_comparison(
            scheduler_data,
            metric_name="avg_response_time",
            show=False,
        )

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_confidence_intervals(self, mock_show):
        """Test plotting confidence intervals."""
        from cgrsp.evaluation.metrics import StatisticalAnalysis, ConfidenceInterval

        viz = PerformanceVisualizer()

        # Create mock statistical analysis
        stats_fifo = StatisticalAnalysis(
            confidence_intervals={
                "avg_response_time": ConfidenceInterval(mean=100.0, lower=95.0, upper=105.0),
            },
            sample_size=30,
        )

        stats_priority = StatisticalAnalysis(
            confidence_intervals={
                "avg_response_time": ConfidenceInterval(mean=90.0, lower=87.0, upper=93.0),
            },
            sample_size=30,
        )

        statistical_analysis = {
            "fifo": stats_fifo,
            "priority": stats_priority,
        }

        # Should not raise exceptions
        viz.plot_confidence_intervals(
            statistical_analysis,
            metric_name="avg_response_time",
            show=False,
        )

        mock_show.assert_not_called()


class TestTimeSeriesVisualizer:
    """Tests for TimeSeriesVisualizer class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        viz = TimeSeriesVisualizer(figsize=(12, 6))
        assert viz.figsize == (12, 6)

    @patch('matplotlib.pyplot.show')
    def test_plot_utilization_timeseries(self, mock_show):
        """Test plotting utilization time series."""
        viz = TimeSeriesVisualizer()

        time_series = [
            (0.0, 0.5),
            (3600.0, 0.7),
            (7200.0, 0.9),
            (10800.0, 0.6),
        ]

        # Should not raise exceptions
        viz.plot_utilization_timeseries(time_series, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_utilization_timeseries_empty(self, mock_show):
        """Test plotting with empty data."""
        viz = TimeSeriesVisualizer()

        # Should handle empty data gracefully
        viz.plot_utilization_timeseries([], show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_queue_length_timeseries(self, mock_show):
        """Test plotting queue length time series."""
        viz = TimeSeriesVisualizer()

        time_series = [
            {"time": 0.0, "queue_length": 5, "idle_nodes": 10, "running_nodes": 5, "unavailable_nodes": 0},
            {"time": 3600.0, "queue_length": 8, "idle_nodes": 7, "running_nodes": 8, "unavailable_nodes": 0},
            {"time": 7200.0, "queue_length": 3, "idle_nodes": 12, "running_nodes": 3, "unavailable_nodes": 0},
        ]

        # Should not raise exceptions
        viz.plot_queue_length_timeseries(time_series, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_node_state_distribution(self, mock_show):
        """Test plotting node state distribution."""
        viz = TimeSeriesVisualizer()

        time_series = [
            {"time": 0.0, "idle_nodes": 10, "running_nodes": 5, "unavailable_nodes": 0},
            {"time": 3600.0, "idle_nodes": 7, "running_nodes": 8, "unavailable_nodes": 0},
            {"time": 7200.0, "idle_nodes": 12, "running_nodes": 3, "unavailable_nodes": 0},
        ]

        # Should not raise exceptions
        viz.plot_node_state_distribution(time_series, show=False)

        mock_show.assert_not_called()


class TestSensitivityVisualizer:
    """Tests for SensitivityVisualizer class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        viz = SensitivityVisualizer(figsize=(10, 8))
        assert viz.figsize == (10, 8)

    @patch('matplotlib.pyplot.show')
    def test_plot_sensitivity_tornado(self, mock_show, sample_sensitivity_result):
        """Test plotting sensitivity tornado chart."""
        viz = SensitivityVisualizer()

        # Should not raise exceptions
        viz.plot_sensitivity_tornado(sample_sensitivity_result, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_sensitivity_tornado_empty(self, mock_show):
        """Test tornado plot with empty data."""
        from cgrsp.evaluation.sensitivity import SensitivityResult

        viz = SensitivityVisualizer()

        empty_result = SensitivityResult()

        # Should handle empty data gracefully
        viz.plot_sensitivity_tornado(empty_result, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_parameter_sweep(self, mock_show):
        """Test plotting parameter sweep."""
        from cgrsp.evaluation.sensitivity import ParameterSweep

        viz = SensitivityVisualizer()

        sweep = ParameterSweep(
            parameter_name="fleet_size",
            parameter_values=[0.5, 1.0, 1.5],
            objective_values=[120.0, 100.0, 80.0],
            qos_values=[1200.0, 1000.0, 800.0],
            cost_values=[600.0, 500.0, 400.0],
            sensitivity_scores=[0.5, 0.0, 0.5],
        )

        # Should not raise exceptions
        viz.plot_parameter_sweep(sweep, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_sensitivity_heatmap(self, mock_show, sample_sensitivity_result):
        """Test plotting sensitivity heatmap."""
        viz = SensitivityVisualizer()

        # Should not raise exceptions
        viz.plot_sensitivity_heatmap(sample_sensitivity_result, show=False)

        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_sensitivity_heatmap_empty(self, mock_show):
        """Test heatmap with empty data."""
        from cgrsp.evaluation.sensitivity import SensitivityResult

        viz = SensitivityVisualizer()

        empty_result = SensitivityResult()

        # Should handle empty data gracefully
        viz.plot_sensitivity_heatmap(empty_result, show=False)

        mock_show.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
