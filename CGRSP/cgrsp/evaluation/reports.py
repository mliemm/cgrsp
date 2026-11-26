"""
Report Generator Module

Generates comprehensive experiment reports with tables, figures,
and statistical significance reporting. Exports to CSV, JSON, and PDF.
"""

import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cgrsp.evaluation.metrics import (
    QoSMetrics,
    OperationalMetrics,
    ObjectiveFunctionDecomposition,
    StatisticalAnalysis,
)
from cgrsp.evaluation.runner import ExperimentResult, ExperimentConfig, ExperimentRunner
from cgrsp.evaluation.sensitivity import SensitivityResult
from cgrsp.evaluation.verification import VerificationReport
from cgrsp.evaluation.visualization import (
    PerformanceVisualizer,
    TimeSeriesVisualizer,
    SensitivityVisualizer,
)


@dataclass
class ExperimentReport:
    """
    Comprehensive experiment report.

    Attributes:
        title: Report title
        timestamp: Generation timestamp
        config: Experiment configuration
        scheduler_results: Results for each scheduler
        comparison: Comparison between schedulers
        sensitivity_results: Sensitivity analysis results
        verification_results: Verification testing results
    """
    title: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[ExperimentConfig] = None
    scheduler_results: Dict[str, ExperimentResult] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    sensitivity_results: Optional[SensitivityResult] = None
    verification_results: Optional[VerificationReport] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "config": self.config.__dict__ if self.config else None,
            "scheduler_results": {
                name: result.to_dict()
                for name, result in self.scheduler_results.items()
            },
            "comparison": self.comparison,
            "sensitivity_results": self.sensitivity_results.to_dict() if self.sensitivity_results else None,
            "verification_results": self.verification_results.to_dict() if self.verification_results else None,
        }


class ReportGenerator:
    """
    Generate comprehensive experiment reports.

    Creates tables, figures, and exports to multiple formats.
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.perf_viz = PerformanceVisualizer()
        self.ts_viz = TimeSeriesVisualizer()
        self.sens_viz = SensitivityVisualizer()

    def generate_report(
        self,
        title: str,
        runner: ExperimentRunner,
        sensitivity_result: Optional[SensitivityResult] = None,
        verification_result: Optional[VerificationReport] = None,
    ) -> ExperimentReport:
        """
        Generate comprehensive experiment report.

        Args:
            title: Report title
            runner: Experiment runner with results
            sensitivity_result: Optional sensitivity analysis results
            verification_result: Optional verification results

        Returns:
            ExperimentReport with all data
        """
        report = ExperimentReport(
            title=title,
            config=runner.config,
            scheduler_results=runner.results,
            comparison=runner.compare_results(),
            sensitivity_results=sensitivity_result,
            verification_results=verification_result,
        )

        return report

    def export_to_json(self, report: ExperimentReport, filepath: Optional[str] = None) -> str:
        """
        Export report to JSON file.

        Args:
            report: Experiment report to export
            filepath: Optional custom filepath (default: output_dir/timestamp.json)

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / f"{report.title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filepath = Path(filepath)

        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        return str(filepath)

    def export_to_csv(self, report: ExperimentReport, filepath: Optional[str] = None) -> str:
        """
        Export metrics to CSV file.

        Args:
            report: Experiment report to export
            filepath: Optional custom filepath (default: output_dir/timestamp.csv)

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / f"{report.title.replace(' ', '_')}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filepath = Path(filepath)

        # Gather metrics for all schedulers
        rows = []

        for scheduler_name, result in report.scheduler_results.items():
            if result.qos_mean and result.operational_mean and result.objective_mean:
                row = {
                    "scheduler": scheduler_name,
                    "avg_waiting_time": result.qos_mean.avg_waiting_time,
                    "avg_response_time": result.qos_mean.avg_response_time,
                    "deadline_adherence_rate": result.qos_mean.deadline_adherence_rate,
                    "p95_response_time": result.qos_mean.p95_response_time,
                    "p99_response_time": result.qos_mean.p99_response_time,
                    "total_cost": result.operational_mean.total_cost,
                    "avg_cost_per_job": result.operational_mean.avg_cost_per_job,
                    "resource_utilization": result.operational_mean.resource_utilization,
                    "cost_efficiency": result.operational_mean.cost_efficiency,
                    "objective_value": result.objective_mean.objective_value,
                    "qos_component": result.objective_mean.qos_component,
                    "cost_component": result.objective_mean.cost_component,
                }
                rows.append(row)

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            if rows:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return str(filepath)

    def export_to_pdf(
        self,
        report: ExperimentReport,
        filepath: Optional[str] = None,
        include_figures: bool = True,
    ) -> str:
        """
        Export report to PDF file.

        Args:
            report: Experiment report to export
            filepath: Optional custom filepath (default: output_dir/timestamp.pdf)
            include_figures: Whether to include figures

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / f"{report.title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        else:
            filepath = Path(filepath)

        with PdfPages(filepath) as pdf:
            # Page 1: Summary
            self._create_summary_page(report, pdf, include_figures)

            # Page 2: Detailed metrics table
            self._create_metrics_table_page(report, pdf)

            # Page 3: Statistical analysis
            if any(r.statistical_analysis for r in report.scheduler_results.values()):
                self._create_statistics_page(report, pdf)

            # Page 4: Sensitivity analysis
            if report.sensitivity_results:
                self._create_sensitivity_page(report, pdf, include_figures)

            # Page 5: Verification results
            if report.verification_results:
                self._create_verification_page(report, pdf)

        return str(filepath)

    def _create_summary_page(self, report: ExperimentReport, pdf: PdfPages, include_figures: bool) -> None:
        """Create summary page for PDF."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        # Title
        fig.suptitle(report.title, fontsize=16, fontweight='bold', y=0.95)

        # Summary text
        summary_text = f"Generated: {report.timestamp}\n\n"

        if report.config:
            summary_text += f"Configuration:\n"
            summary_text += f"  Episodes: {report.config.num_episodes}\n"
            summary_text += f"  Duration: {report.config.simulation_duration}s\n"
            summary_text += f"  Alpha (QoS weight): {report.config.alpha_weight}\n\n"

        summary_text += "Schedulers Tested:\n"
        for name in report.scheduler_results.keys():
            summary_text += f"  - {name}\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add performance comparison figure if requested
        if include_figures and len(report.scheduler_results) > 1:
            fig, ax = plt.subplots(figsize=(11, 8.5))

            metrics = ["avg_response_time", "deadline_adherence_rate", "total_cost", "resource_utilization"]
            self.perf_viz.plot_metric_comparison(
                report.scheduler_results,
                metrics,
                title="Scheduler Performance Comparison",
                show=False,
            )

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    def _create_metrics_table_page(self, report: ExperimentReport, pdf: PdfPages) -> None:
        """Create detailed metrics table page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        fig.suptitle("Detailed Metrics", fontsize=16, fontweight='bold', y=0.95)

        # Create table data
        table_data = []
        headers = ["Scheduler", "Avg Wait", "Avg Response", "Deadline Rate",
                  "Total Cost", "Avg Cost/Job", "Utilization", "Objective Z"]

        for name, result in report.scheduler_results.items():
            if result.qos_mean and result.operational_mean and result.objective_mean:
                row = [
                    name,
                    f"{result.qos_mean.avg_waiting_time:.1f}s",
                    f"{result.qos_mean.avg_response_time:.1f}s",
                    f"{result.qos_mean.deadline_adherence_rate*100:.1f}%",
                    f"${result.operational_mean.total_cost:.2f}",
                    f"${result.operational_mean.avg_cost_per_job:.2f}",
                    f"{result.operational_mean.resource_utilization*100:.1f}%",
                    f"{result.objective_mean.objective_value:.4f}",
                ]
                table_data.append(row)

        # Create table
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center',
                           bbox=[0.1, 0.2, 0.8, 0.6])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header row
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_statistics_page(self, report: ExperimentReport, pdf: PdfPages) -> None:
        """Create statistical analysis page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        fig.suptitle("Statistical Analysis", fontsize=16, fontweight='bold', y=0.95)

        # Gather statistical results
        stats_text = ""

        for name, result in report.scheduler_results.items():
            if result.statistical_analysis and result.statistical_analysis.confidence_intervals:
                stats_text += f"\n{name}:\n"
                stats_text += f"  Sample size: n={result.statistical_analysis.sample_size}\n\n"

                stats_text += "  95% Confidence Intervals:\n"
                for metric_name, ci in result.statistical_analysis.confidence_intervals.items():
                    stats_text += f"    {metric_name}:\n"
                    stats_text += f"      Mean: {ci.mean:.4f}\n"
                    stats_text += f"      CI: [{ci.lower:.4f}, {ci.upper:.4f}]\n"

                stats_text += "\n"

        # Paired tests
        if report.comparison:
            stats_text += "\nScheduler Comparisons:\n"
            for comparison_name, comp_data in report.comparison.items():
                stats_text += f"\n{comparison_name}:\n"
                if "paired_tests" in comp_data:
                    for metric_name, test_result in comp_data["paired_tests"].items():
                        sig = "Significant" if test_result["is_significant"] else "Not significant"
                        stats_text += f"  {metric_name}:\n"
                        stats_text += f"    p-value: {test_result['p_value']:.4f} ({sig})\n"
                        if test_result.get("effect_size"):
                            stats_text += f"    Effect size: {test_result['effect_size']:.4f}\n"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_sensitivity_page(self, report: ExperimentReport, pdf: PdfPages, include_figures: bool) -> None:
        """Create sensitivity analysis page."""
        if not report.sensitivity_results:
            return

        # Text summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        fig.suptitle("Sensitivity Analysis", fontsize=16, fontweight='bold', y=0.95)

        sens_text = f"Baseline Objective: {report.sensitivity_results.baseline_objective:.4f}\n\n"

        if report.sensitivity_results.high_sensitivity_params:
            sens_text += "High-Sensitivity Parameters (|S(θ)| > 1):\n"
            for param in report.sensitivity_results.high_sensitivity_params:
                sens_text += f"  - {param}\n"
        else:
            sens_text += "No high-sensitivity parameters detected.\n"

        sens_text += "\nParameter Ranking:\n"
        for param, score in report.sensitivity_results.sensitivity_ranking:
            sens_text += f"  {param}: {score:.4f}\n"

        ax.text(0.05, 0.95, sens_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add sensitivity figure if requested
        if include_figures:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            self.sens_viz.plot_sensitivity_tornado(
                report.sensitivity_results,
                show=False,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    def _create_verification_page(self, report: ExperimentReport, pdf: PdfPages) -> None:
        """Create verification results page."""
        if not report.verification_results:
            return

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        fig.suptitle("Verification Results", fontsize=16, fontweight='bold', y=0.95)

        status = "PASSED" if report.verification_results.all_passed else "FAILED"
        verif_text = f"Overall Status: {status}\n"
        verif_text += f"Total Violations: {report.verification_results.total_violations}\n\n"

        if report.verification_results.constraint_results:
            verif_text += "Constraint Checks:\n"
            for constraint_name, result in report.verification_results.constraint_results.items():
                verif_text += f"  {constraint_name}: {result.get('status', 'unknown')}\n"
                if result.get('description'):
                    verif_text += f"    {result['description']}\n"

        if report.verification_results.violations:
            verif_text += "\nViolations:\n"
            for i, violation in enumerate(report.verification_results.violations[:10], 1):
                verif_text += f"\n{i}. {violation.constraint_name}\n"
                verif_text += f"   Type: {violation.violation_type}\n"
                verif_text += f"   Severity: {violation.severity}\n"
                verif_text += f"   Description: {violation.description}\n"

            if len(report.verification_results.violations) > 10:
                verif_text += f"\n... and {len(report.verification_results.violations) - 10} more violations\n"

        ax.text(0.05, 0.95, verif_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def generate_all_exports(
        self,
        report: ExperimentReport,
        base_filename: Optional[str] = None,
        include_figures: bool = True,
    ) -> Dict[str, str]:
        """
        Generate all export formats for a report.

        Args:
            report: Experiment report to export
            base_filename: Base filename (without extension)
            include_figures: Whether to include figures in PDF

        Returns:
            Dictionary of format to filepath
        """
        if base_filename is None:
            base_filename = self.output_dir / report.title.replace(' ', '_')

        exports = {}

        # JSON
        json_path = self.export_to_json(report, f"{base_filename}.json")
        exports["json"] = json_path

        # CSV
        csv_path = self.export_to_csv(report, f"{base_filename}.csv")
        exports["csv"] = csv_path

        # PDF
        pdf_path = self.export_to_pdf(report, f"{base_filename}.pdf", include_figures)
        exports["pdf"] = pdf_path

        return exports

    def print_report_summary(self, report: ExperimentReport) -> None:
        """
        Print a summary of the report to console.

        Args:
            report: Experiment report to summarize
        """
        print("\n" + "="*80)
        print(f"REPORT: {report.title}")
        print("="*80)
        print(f"Generated: {report.timestamp}")

        if report.config:
            print(f"\nConfiguration:")
            print(f"  Episodes per scheduler: {report.config.num_episodes}")
            print(f"  Simulation duration: {report.config.simulation_duration}s")
            print(f"  QoS weight (α): {report.config.alpha_weight}")

        print(f"\nSchedulers: {', '.join(report.scheduler_results.keys())}")

        print(f"\nPerformance Summary:")
        for name, result in report.scheduler_results.items():
            if result.qos_mean and result.operational_mean:
                print(f"\n  {name}:")
                print(f"    Avg response time: {result.qos_mean.avg_response_time:.1f}s")
                print(f"    Deadline adherence: {result.qos_mean.deadline_adherence_rate*100:.1f}%")
                print(f"    Total cost: ${result.operational_mean.total_cost:.2f}")
                print(f"    Objective Z: {result.objective_mean.objective_value:.4f}" if result.objective_mean else "")

        if report.sensitivity_results:
            print(f"\nSensitivity Analysis:")
            print(f"  High-sensitivity params: {', '.join(report.sensitivity_results.high_sensitivity_params) or 'None'}")

        if report.verification_results:
            status = "PASSED" if report.verification_results.all_passed else "FAILED"
            print(f"\nVerification: {status}")
            print(f"  Violations found: {report.verification_results.total_violations}")

        print("\n" + "="*80)
