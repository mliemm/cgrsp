"""
Tests for report generator module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cgrsp.evaluation.reports import (
    ExperimentReport,
    ReportGenerator,
)


@pytest.fixture
def sample_report():
    """Create sample experiment report."""
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

    report = ExperimentReport(
        title="Test Experiment",
        config=config,
        scheduler_results={"fifo": result},
    )

    return report


class TestExperimentReport:
    """Tests for ExperimentReport dataclass."""

    def test_report_creation(self, sample_report):
        """Test report creation."""
        assert sample_report.title == "Test Experiment"
        assert sample_report.config is not None
        assert len(sample_report.scheduler_results) == 1

    def test_report_to_dict(self, sample_report):
        """Test report serialization."""
        data = sample_report.to_dict()

        assert isinstance(data, dict)
        assert data["title"] == "Test Experiment"
        assert "scheduler_results" in data
        assert len(data["scheduler_results"]) == 1


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            assert generator.output_dir == Path(tmpdir)
            assert generator.perf_viz is not None
            assert generator.ts_viz is not None
            assert generator.sens_viz is not None

    def test_export_to_json(self, sample_report):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            filepath = generator.export_to_json(sample_report)

            assert Path(filepath).exists()

            # Verify content
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data["title"] == "Test Experiment"
            assert "scheduler_results" in data

    def test_export_to_csv(self, sample_report):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            filepath = generator.export_to_csv(sample_report)

            assert Path(filepath).exists()

            # Verify content
            import csv
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["scheduler"] == "fifo"

    @patch('cgrsp.evaluation.reports.PdfPages')
    def test_export_to_pdf(self, mock_pdf_pages, sample_report):
        """Test PDF export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            # Mock PdfPages context manager
            mock_context = MagicMock()
            mock_pdf_pages.return_value.__enter__.return_value = mock_context

            filepath = generator.export_to_pdf(sample_report, include_figures=False)

            # Should call PdfPages
            mock_pdf_pages.assert_called_once()

    def test_generate_all_exports(self, sample_report):
        """Test generating all export formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            # Mock PDF to avoid matplotlib issues and ensure the file is created
            with patch('cgrsp.evaluation.reports.PdfPages') as mock_pdf:
                mock_pdf.side_effect = lambda filepath: Path(filepath).touch() or MagicMock()
                exports = generator.generate_all_exports(
                    sample_report,
                    base_filename=str(Path(tmpdir) / "test_report"),
                    include_figures=False,
                )

            assert "json" in exports
            assert "csv" in exports
            assert "pdf" in exports

            # Verify files exist
            for format_name, filepath in exports.items():
                assert Path(filepath).exists()

    def test_generate_report(self):
        """Test report generation from runner."""
        from cgrsp.evaluation.runner import ExperimentRunner, ExperimentConfig
        from cgrsp.config import ResourceType

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)

            config = ExperimentConfig(
                fleet_config=[(ResourceType.RTX_3090, 4)],
                num_episodes=2,
            )

            runner = ExperimentRunner(config)
            # Don't actually run, just create mock results

            report = generator.generate_report(
                title="Test",
                runner=runner,
            )

            assert report.title == "Test"
            assert report.config == config

    @patch('builtins.print')
    def test_print_report_summary(self, mock_print, sample_report):
        """Test printing report summary."""
        generator = ReportGenerator()

        generator.print_report_summary(sample_report)

        # Verify print was called
        assert mock_print.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
