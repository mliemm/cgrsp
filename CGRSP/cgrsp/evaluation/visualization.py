"""
Visualization Module

Creates charts and plots for experiment results including
performance comparisons, time-series, and sensitivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from cgrsp.evaluation.metrics import QoSMetrics, OperationalMetrics, StatisticalAnalysis
from cgrsp.evaluation.runner import ExperimentResult, ExperimentConfig
from cgrsp.evaluation.sensitivity import SensitivityResult, ParameterSweep


class PerformanceVisualizer:
    """
    Create performance comparison charts.

    Generates bar plots, box plots, and comparison charts
    for scheduler performance metrics.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_metric_comparison(
        self,
        results: Dict[str, ExperimentResult],
        metrics: List[str],
        title: str = "Scheduler Comparison",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot bar chart comparing schedulers on multiple metrics.

        Args:
            results: Dictionary of scheduler names to results
            metrics: List of metric names to plot
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        scheduler_names = list(results.keys())
        x = np.arange(len(scheduler_names))
        width = 0.8 / len(metrics)

        # Get metric values
        metric_values = {}
        for metric in metrics:
            values = []
            for name in scheduler_names:
                result = results[name]

                # Try to get metric from different sources
                value = None

                # Check qos_mean
                if result.qos_mean:
                    value = getattr(result.qos_mean, metric, None)

                # Check operational_mean
                if value is None and result.operational_mean:
                    value = getattr(result.operational_mean, metric, None)

                # Check objective_mean
                if value is None and result.objective_mean:
                    value = getattr(result.objective_mean, metric, None)

                values.append(value if value is not None else 0.0)

            metric_values[metric] = values

        # Plot bars
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        bars = []

        for i, (metric, values) in enumerate(metric_values.items()):
            offset = (i - len(metrics)/2 + 0.5) * width
            bar = ax.bar(x + offset, values, width, label=metric, color=colors[i])
            bars.append(bar)

        # Formatting
        ax.set_xlabel('Scheduler', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(scheduler_names)
        ax.legend()

        # Add value labels on bars
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_box_comparison(
        self,
        scheduler_data: Dict[str, List[float]],
        metric_name: str,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot box plots comparing metric distributions across schedulers.

        Args:
            scheduler_data: Dictionary of scheduler names to metric values
            metric_name: Name of metric being plotted
            title: Chart title (auto-generated if None)
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Prepare data
        data = [scheduler_data[name] for name in scheduler_data.keys()]
        labels = list(scheduler_data.keys())

        # Create box plot
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Formatting
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f"{metric_name.replace('_', ' ').title()} Distribution by Scheduler", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confidence_intervals(
        self,
        statistical_analysis: Dict[str, StatisticalAnalysis],
        metric_name: str,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot 95% confidence intervals for metric across schedulers.

        Args:
            statistical_analysis: Dictionary of scheduler names to statistical analysis
            metric_name: Name of metric to plot
            title: Chart title (auto-generated if None)
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        scheduler_names = list(statistical_analysis.keys())

        # Extract confidence intervals
        means = []
        lower_bounds = []
        upper_bounds = []

        for name in scheduler_names:
            stats = statistical_analysis[name]

            if metric_name in stats.confidence_intervals:
                ci = stats.confidence_intervals[metric_name]
                means.append(ci.mean)
                lower_bounds.append(ci.lower)
                upper_bounds.append(ci.upper)
            else:
                means.append(0.0)
                lower_bounds.append(0.0)
                upper_bounds.append(0.0)

        x = np.arange(len(scheduler_names))

        # Plot error bars
        ax.errorbar(x, means, yerr=[np.array(means) - np.array(lower_bounds),
                                    np.array(upper_bounds) - np.array(means)],
                   fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(scheduler_names)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f"95% Confidence Intervals: {metric_name.replace('_', ' ').title()}", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class TimeSeriesVisualizer:
    """
    Create time-series visualizations.

    Plots utilization, queue length, and other metrics over time.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_utilization_timeseries(
        self,
        time_series_data: List[Tuple[float, float]],
        title: str = "Resource Utilization Over Time",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot resource utilization over time.

        Args:
            time_series_data: List of (time, utilization) tuples
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not time_series_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return

        times = [t[0] / 3600 for t in time_series_data]  # Convert to hours
        utilizations = [t[1] * 100 for t in time_series_data]  # Convert to percentage

        ax.plot(times, utilizations, linewidth=2, color='steelblue')
        ax.fill_between(times, utilizations, alpha=0.3, color='steelblue')

        # Formatting
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Utilization (%)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_queue_length_timeseries(
        self,
        time_series_data: List[Dict],
        title: str = "Queue Length Over Time",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot queue length over time.

        Args:
            time_series_data: List of time-series state dictionaries
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not time_series_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return

        times = [d["time"] / 3600 for d in time_series_data]  # Convert to hours
        queue_lengths = [d["queue_length"] for d in time_series_data]

        ax.plot(times, queue_lengths, linewidth=2, color='coral')

        # Formatting
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Queue Length', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_node_state_distribution(
        self,
        time_series_data: List[Dict],
        title: str = "Node State Distribution Over Time",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of node states over time.

        Args:
            time_series_data: List of time-series state dictionaries
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not time_series_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return

        times = [d["time"] / 3600 for d in time_series_data]
        idle = [d["idle_nodes"] for d in time_series_data]
        running = [d["running_nodes"] for d in time_series_data]
        unavailable = [d["unavailable_nodes"] for d in time_series_data]

        ax.stackplot(times, idle, running, unavailable,
                    labels=['Idle', 'Running', 'Unavailable'],
                    alpha=0.8, colors=['lightgreen', 'steelblue', 'lightcoral'])

        # Formatting
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class SensitivityVisualizer:
    """
    Create sensitivity analysis visualizations.

    Generates heatmaps, tornado plots, and parameter sensitivity charts.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_sensitivity_tornado(
        self,
        sensitivity_result: SensitivityResult,
        title: str = "Parameter Sensitivity Tornado Plot",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create tornado plot showing parameter sensitivities.

        Args:
            sensitivity_result: Sensitivity analysis results
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get ranking
        ranking = sensitivity_result.sensitivity_ranking
        if not ranking:
            ax.text(0.5, 0.5, 'No sensitivity data available', ha='center', va='center')
            return

        param_names = [p[0] for p in ranking]
        sensitivities = [p[1] for p in ranking]

        # Create horizontal bar chart
        y_pos = np.arange(len(param_names))
        colors = ['red' if s > 1.0 else 'orange' if s > 0.5 else 'green' for s in sensitivities]

        ax.barh(y_pos, sensitivities, color=colors, alpha=0.7)

        # Add threshold line
        ax.axvline(x=1.0, color='darkred', linestyle='--', linewidth=2, label='High sensitivity threshold')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Sensitivity |S(θ)|', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_parameter_sweep(
        self,
        sweep_result: ParameterSweep,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot parameter sweep results.

        Args:
            sweep_result: Parameter sweep results
            title: Chart title (auto-generated if None)
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))

        param_values = sweep_result.parameter_values

        # Objective function vs parameter
        ax1.plot(param_values, sweep_result.objective_values, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xlabel(f'{sweep_result.parameter_name}'.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel('Objective Value Z', fontsize=12)
        ax1.set_title(f'Objective Function vs {sweep_result.parameter_name}'.replace('_', ' ').title(), fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Sensitivity score
        ax2.plot(param_values, sweep_result.sensitivity_scores, 's-', linewidth=2, markersize=8, color='coral')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='High sensitivity threshold')
        ax2.set_xlabel(f'{sweep_result.parameter_name}'.replace('_', ' ').title(), fontsize=12)
        ax2.set_ylabel('Sensitivity |S(θ)|', fontsize=12)
        ax2.set_title(f'Sensitivity Score vs {sweep_result.parameter_name}'.replace('_', ' ').title(), fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=0.995)
        else:
            fig.suptitle(f'Parameter Sweep: {sweep_result.parameter_name}'.replace('_', ' ').title(),
                        fontsize=14, y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_sensitivity_heatmap(
        self,
        sensitivity_result: SensitivityResult,
        title: str = "Parameter Sensitivity Heatmap",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sensitivity heatmap across all parameters.

        Args:
            sensitivity_result: Sensitivity analysis results
            title: Chart title
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        import matplotlib as mpl

        fig, ax = plt.subplots(figsize=self.figsize)

        param_names = list(sensitivity_result.parameter_sweeps.keys())

        if not param_names:
            ax.text(0.5, 0.5, 'No sensitivity data available', ha='center', va='center')
            return

        # Get max sensitivity for each parameter
        max_sensitivities = [
            max(sensitivity_result.parameter_sweeps[name].sensitivity_scores)
            for name in param_names
        ]

        # Create heatmap
        data = np.array(max_sensitivities).reshape(1, -1)

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks([0])
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticklabels([''])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Max Sensitivity |S(θ)|', rotation=270, labelpad=20)

        # Add text annotations
        for i, val in enumerate(max_sensitivities):
            text_color = 'black' if val < 0.5 else 'white'
            ax.text(i, 0, f'{val:.2f}', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')

        ax.set_title(title, fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()
