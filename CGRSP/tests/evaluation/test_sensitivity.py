"""
Tests for sensitivity analysis module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from cgrsp.evaluation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    ParameterSweep,
)


@pytest.fixture
def sample_config():
    """Create sample experiment configuration."""
    from cgrsp.evaluation.runner import ExperimentConfig
    from cgrsp.config import ResourceType

    return ExperimentConfig(
        fleet_config=[(ResourceType.RTX_3090, 4), (ResourceType.RTX_A4000, 8)],
        num_episodes=5,
        simulation_duration=3600.0,
        alpha_weight=0.5,
    )


@pytest.fixture
def mock_scheduler():
    """Create mock scheduler function."""
    def scheduler(state, simulator):
        pass
    return scheduler


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    def test_initialization(self, sample_config, mock_scheduler):
        """Test analyzer initialization."""
        analyzer = SensitivityAnalyzer(sample_config, mock_scheduler)

        assert analyzer.base_config == sample_config
        assert analyzer.scheduler_func == mock_scheduler

    def test_compute_sensitivity_score(self, sample_config, mock_scheduler):
        """Test sensitivity score calculation."""
        analyzer = SensitivityAnalyzer(sample_config, mock_scheduler)

        score = analyzer.compute_sensitivity_score(
            baseline_value=100.0,
            current_value=110.0,
            baseline_param=1.0,
            current_param=1.2,
        )

        # S(θ) = (10/100) / (0.2/1.0) = 0.1 / 0.2 = 0.5
        assert abs(score - 0.5) < 0.01

    def test_compute_sensitivity_score_zero_baseline(self, sample_config, mock_scheduler):
        """Test sensitivity score with zero baseline."""
        analyzer = SensitivityAnalyzer(sample_config, mock_scheduler)

        score = analyzer.compute_sensitivity_score(
            baseline_value=0.0,
            current_value=10.0,
            baseline_param=1.0,
            current_param=2.0,
        )

        assert score == 0.0

    def test_compute_sensitivity_score_no_change(self, sample_config, mock_scheduler):
        """Test sensitivity score with no parameter change."""
        analyzer = SensitivityAnalyzer(sample_config, mock_scheduler)

        score = analyzer.compute_sensitivity_score(
            baseline_value=100.0,
            current_value=110.0,
            baseline_param=1.0,
            current_param=1.0,  # No change
        )

        assert score == 0.0


class TestParameterSweep:
    """Tests for ParameterSweep dataclass."""

    def test_parameter_sweep_creation(self):
        """Test parameter sweep creation."""
        sweep = ParameterSweep(
            parameter_name="fleet_size",
            parameter_values=[0.5, 1.0, 1.5],
            objective_values=[100.0, 80.0, 60.0],
            qos_values=[1000.0, 800.0, 600.0],
            cost_values=[500.0, 400.0, 300.0],
            sensitivity_scores=[0.5, 0.0, 0.5],
        )

        assert sweep.parameter_name == "fleet_size"
        assert len(sweep.parameter_values) == 3
        assert len(sweep.objective_values) == 3

    def test_parameter_sweep_to_dict(self):
        """Test parameter sweep serialization."""
        sweep = ParameterSweep(
            parameter_name="test",
            parameter_values=[1.0, 2.0],
            objective_values=[100.0, 90.0],
            qos_values=[1000.0, 900.0],
            cost_values=[500.0, 450.0],
            sensitivity_scores=[0.2, 0.1],
        )

        data = sweep.to_dict()

        assert isinstance(data, dict)
        assert data["parameter_name"] == "test"
        assert len(data["parameter_values"]) == 2


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_sensitivity_result_creation(self):
        """Test sensitivity result creation."""
        sweep = ParameterSweep(
            parameter_name="test",
            parameter_values=[1.0],
            objective_values=[100.0],
            qos_values=[1000.0],
            cost_values=[500.0],
            sensitivity_scores=[0.5],
        )

        result = SensitivityResult(
            parameter_sweeps={"test": sweep},
            high_sensitivity_params=["test"],
            sensitivity_ranking=[("test", 0.5)],
            baseline_objective=100.0,
        )

        assert len(result.parameter_sweeps) == 1
        assert len(result.high_sensitivity_params) == 1
        assert result.baseline_objective == 100.0

    def test_sensitivity_result_to_dict(self):
        """Test sensitivity result serialization."""
        sweep = ParameterSweep(
            parameter_name="test",
            parameter_values=[1.0],
            objective_values=[100.0],
            qos_values=[1000.0],
            cost_values=[500.0],
            sensitivity_scores=[0.5],
        )

        result = SensitivityResult(
            parameter_sweeps={"test": sweep},
            baseline_objective=100.0,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert "parameter_sweeps" in data
        assert "high_sensitivity_params" in data
        assert data["baseline_objective"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
