"""
Sensitivity Analysis Module

Analyzes parameter sensitivity and performs parameter sweeps
to identify high-sensitivity parameters in the CGRSP system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from copy import deepcopy

from cgrsp.simulator.core import Simulator
from cgrsp.config import ResourceType, TemporalParameters, AvailabilityParameters
from cgrsp.evaluation.runner import ExperimentConfig, _run_single_episode
from cgrsp.evaluation.metrics import PerformanceMetrics, QoSMetrics, OperationalMetrics


@dataclass
class ParameterSweep:
    """
    Results from sweeping a single parameter.

    Attributes:
        parameter_name: Name of parameter swept
        parameter_values: List of parameter values tested
        objective_values: Objective function value at each point
        qos_values: QoS metric at each point
        cost_values: Cost metric at each point
        sensitivity_scores: Sensitivity score at each point
    """
    parameter_name: str
    parameter_values: List[float]
    objective_values: List[float]
    qos_values: List[float]
    cost_values: List[float]
    sensitivity_scores: List[float]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "objective_values": self.objective_values,
            "qos_values": self.qos_values,
            "cost_values": self.cost_values,
            "sensitivity_scores": self.sensitivity_scores,
        }


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.

    Attributes:
        parameter_sweeps: Dictionary of parameter name to sweep results
        high_sensitivity_params: List of parameters with |S(θ)| > 1
        sensitivity_ranking: Parameters ranked by sensitivity
        baseline_objective: Objective value at baseline parameters
    """
    parameter_sweeps: Dict[str, ParameterSweep] = field(default_factory=dict)
    high_sensitivity_params: List[str] = field(default_factory=list)
    sensitivity_ranking: List[tuple[str, float]] = field(default_factory=list)
    baseline_objective: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "parameter_sweeps": {
                k: v.to_dict() for k, v in self.parameter_sweeps.items()
            },
            "high_sensitivity_params": self.high_sensitivity_params,
            "sensitivity_ranking": [
                {"param": p, "sensitivity": s} for p, s in self.sensitivity_ranking
            ],
            "baseline_objective": self.baseline_objective,
        }


class SensitivityAnalyzer:
    """
    Analyze parameter sensitivity in the CGRSP system.

    Sensitivity metric: S(θ) = (ΔZ/Z₀) / (Δθ/θ₀)
    where Z is objective value, θ is parameter value, subscript 0 is baseline.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        scheduler_func: Callable,
        job_generator: Optional[Callable] = None,
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            base_config: Base experiment configuration
            scheduler_func: Scheduler function to test
            job_generator: Optional job generator
        """
        self.base_config = base_config
        self.scheduler_func = scheduler_func
        self.job_generator = job_generator
        self.metrics_calculator = PerformanceMetrics(alpha_weight=base_config.alpha_weight)

    def compute_sensitivity_score(
        self,
        baseline_value: float,
        current_value: float,
        baseline_param: float,
        current_param: float,
    ) -> float:
        """
        Compute sensitivity score S(θ).

        S(θ) = (ΔZ/Z₀) / (Δθ/θ₀)

        Args:
            baseline_value: Objective value at baseline (Z₀)
            current_value: Objective value at current parameter (Z)
            baseline_param: Baseline parameter value (θ₀)
            current_param: Current parameter value (θ)

        Returns:
            Sensitivity score S(θ)
        """
        # Avoid division by zero
        if baseline_value == 0 or baseline_param == 0:
            return 0.0

        delta_z = current_value - baseline_value
        delta_theta = current_param - baseline_param

        if delta_theta == 0:
            return 0.0

        relative_change_z = delta_z / baseline_value
        relative_change_theta = delta_theta / baseline_param

        if relative_change_theta == 0:
            return 0.0

        return relative_change_z / relative_change_theta

    def sweep_parameter(
        self,
        parameter_name: str,
        parameter_values: List[float],
        param_modifier: Callable[[ExperimentConfig, float], ExperimentConfig],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Sweep a parameter across a range of values.

        Args:
            parameter_name: Name of parameter to sweep
            parameter_values: List of parameter values to test
            param_modifier: Function that modifies config with new parameter value
            num_trials: Number of trials per parameter value

        Returns:
            ParameterSweep with results
        """
        objective_values = []
        qos_values = []
        cost_values = []
        sensitivity_scores = []

        # Run baseline
        baseline_results = []
        for i in range(num_trials):
            seed = self.base_config.random_seeds[i] if i < len(self.base_config.random_seeds) else i

            episode = _run_single_episode(
                scheduler_func=self.scheduler_func,
                scheduler_name="baseline",
                config=self.base_config,
                episode_id=i,
                random_seed=seed,
                job_generator=self.job_generator,
            )
            baseline_results.append(episode)

        baseline_objective = np.mean([ep.objective_decomposition.objective_value for ep in baseline_results])
        baseline_qos = np.mean([ep.qos_metrics.avg_response_time for ep in baseline_results])
        baseline_cost = np.mean([ep.operational_metrics.total_cost for ep in baseline_results])

        # Sweep parameter
        for param_value in parameter_values:
            # Modify config
            modified_config = param_modifier(self.base_config, param_value)

            # Run trials
            trial_results = []
            for i in range(num_trials):
                seed = modified_config.random_seeds[i] if i < len(modified_config.random_seeds) else i

                episode = _run_single_episode(
                    scheduler_func=self.scheduler_func,
                    scheduler_name=f"sweep_{parameter_name}",
                    config=modified_config,
                    episode_id=i,
                    random_seed=seed,
                    job_generator=self.job_generator,
                )
                trial_results.append(episode)

            # Average metrics
            avg_objective = np.mean([ep.objective_decomposition.objective_value for ep in trial_results])
            avg_qos = np.mean([ep.qos_metrics.avg_response_time for ep in trial_results])
            avg_cost = np.mean([ep.operational_metrics.total_cost for ep in trial_results])

            # Compute sensitivity
            sensitivity = self.compute_sensitivity_score(
                baseline_value=baseline_objective,
                current_value=avg_objective,
                baseline_param=parameter_values[len(parameter_values) // 2],  # Use middle value as baseline
                current_param=param_value,
            )

            objective_values.append(avg_objective)
            qos_values.append(avg_qos)
            cost_values.append(avg_cost)
            sensitivity_scores.append(abs(sensitivity))

        return ParameterSweep(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            objective_values=objective_values,
            qos_values=qos_values,
            cost_values=cost_values,
            sensitivity_scores=sensitivity_scores,
        )

    def analyze_fleet_size_sensitivity(
        self,
        size_multipliers: List[float],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Analyze sensitivity to fleet size.

        Args:
            size_multipliers: List of fleet size multipliers (e.g., [0.5, 0.75, 1.0, 1.25, 1.5])
            num_trials: Number of trials per multiplier

        Returns:
            ParameterSweep with results
        """

        def modify_fleet_size(config: ExperimentConfig, multiplier: float) -> ExperimentConfig:
            """Modify fleet size by multiplier."""
            new_config = deepcopy(config)
            new_config.fleet_config = [
                (rt, int(count * multiplier))
                for rt, count in config.fleet_config
            ]
            return new_config

        return self.sweep_parameter(
            parameter_name="fleet_size_multiplier",
            parameter_values=size_multipliers,
            param_modifier=modify_fleet_size,
            num_trials=num_trials,
        )

    def analyze_workload_intensity_sensitivity(
        self,
        intensity_multipliers: List[float],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Analyze sensitivity to workload intensity.

        Args:
            intensity_multipliers: List of workload intensity multipliers
            num_trials: Number of trials per multiplier

        Returns:
            ParameterSweep with results
        """
        # This requires modifying the job generator
        # For now, return a placeholder
        return ParameterSweep(
            parameter_name="workload_intensity",
            parameter_values=intensity_multipliers,
            objective_values=[0.0] * len(intensity_multipliers),
            qos_values=[0.0] * len(intensity_multipliers),
            cost_values=[0.0] * len(intensity_multipliers),
            sensitivity_scores=[0.0] * len(intensity_multipliers),
        )

    def analyze_availability_sensitivity(
        self,
        availability_rates: List[float],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Analyze sensitivity to node availability rate.

        Args:
            availability_rates: List of availability rates (0-1)
            num_trials: Number of trials per rate

        Returns:
            ParameterSweep with results
        """

        def modify_availability(config: ExperimentConfig, rate: float) -> ExperimentConfig:
            """Modify availability parameters."""
            new_config = deepcopy(config)
            if new_config.availability_params is None:
                new_config.availability_params = AvailabilityParameters()

            # Scale transition rates inversely with availability
            # Higher availability = lower transition to unavailable
            base_rate = 1.0
            scale_factor = (1.0 - rate) / (1.0 - 0.9)  # Normalize around 0.9 baseline

            new_config.availability_params.lambda_customer_base *= scale_factor
            new_config.availability_params.lambda_maintenance_base *= scale_factor
            new_config.availability_params.lambda_failure *= scale_factor

            return new_config

        return self.sweep_parameter(
            parameter_name="availability_rate",
            parameter_values=availability_rates,
            param_modifier=modify_availability,
            num_trials=num_trials,
        )

    def analyze_qos_weight_sensitivity(
        self,
        alpha_weights: List[float],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Analyze sensitivity to QoS weight (α) in objective function.

        Args:
            alpha_weights: List of α values in [0, 1]
            num_trials: Number of trials per weight

        Returns:
            ParameterSweep with results
        """

        def modify_alpha(config: ExperimentConfig, alpha: float) -> ExperimentConfig:
            """Modify QoS weight."""
            new_config = deepcopy(config)
            new_config.alpha_weight = alpha
            return new_config

        return self.sweep_parameter(
            parameter_name="qos_weight_alpha",
            parameter_values=alpha_weights,
            param_modifier=modify_alpha,
            num_trials=num_trials,
        )

    def analyze_deadline_penalty_sensitivity(
        self,
        penalty_multipliers: List[float],
        num_trials: int = 5,
    ) -> ParameterSweep:
        """
        Analyze sensitivity to deadline penalty weight.

        Args:
            penalty_multipliers: List of penalty multipliers
            num_trials: Number of trials per multiplier

        Returns:
            ParameterSweep with results
        """
        # This is embedded in the objective computation
        # For now, return a placeholder
        return ParameterSweep(
            parameter_name="deadline_penalty",
            parameter_values=penalty_multipliers,
            objective_values=[0.0] * len(penalty_multipliers),
            qos_values=[0.0] * len(penalty_multipliers),
            cost_values=[0.0] * len(penalty_multipliers),
            sensitivity_scores=[0.0] * len(penalty_multipliers),
        )

    def run_full_analysis(
        self,
        fleet_size_multipliers: Optional[List[float]] = None,
        availability_rates: Optional[List[float]] = None,
        alpha_weights: Optional[List[float]] = None,
        num_trials: int = 5,
    ) -> SensitivityResult:
        """
        Run full sensitivity analysis across all parameters.

        Args:
            fleet_size_multipliers: Fleet size multipliers to test
            availability_rates: Availability rates to test
            alpha_weights: QoS weights to test
            num_trials: Number of trials per parameter value

        Returns:
            SensitivityResult with all sweeps
        """
        # Default parameter ranges
        if fleet_size_multipliers is None:
            fleet_size_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        if availability_rates is None:
            availability_rates = [0.85, 0.9, 0.95, 0.975, 0.99]
        if alpha_weights is None:
            alpha_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

        result = SensitivityResult()

        # Run baseline
        baseline_episode = _run_single_episode(
            scheduler_func=self.scheduler_func,
            scheduler_name="baseline",
            config=self.base_config,
            episode_id=0,
            random_seed=self.base_config.random_seeds[0],
            job_generator=self.job_generator,
        )
        result.baseline_objective = baseline_episode.objective_decomposition.objective_value

        # Fleet size sensitivity
        print("Analyzing fleet size sensitivity...")
        fleet_sweep = self.analyze_fleet_size_sensitivity(fleet_size_multipliers, num_trials)
        result.parameter_sweeps["fleet_size"] = fleet_sweep

        # Availability sensitivity
        print("Analyzing availability sensitivity...")
        availability_sweep = self.analyze_availability_sensitivity(availability_rates, num_trials)
        result.parameter_sweeps["availability"] = availability_sweep

        # QoS weight sensitivity
        print("Analyzing QoS weight sensitivity...")
        qos_sweep = self.analyze_qos_weight_sensitivity(alpha_weights, num_trials)
        result.parameter_sweeps["qos_weight"] = qos_sweep

        # Identify high-sensitivity parameters (|S(θ)| > 1)
        for param_name, sweep in result.parameter_sweeps.items():
            max_sensitivity = max(sweep.sensitivity_scores)
            if max_sensitivity > 1.0:
                result.high_sensitivity_params.append(param_name)

        # Rank parameters by sensitivity
        sensitivity_ranking = []
        for param_name, sweep in result.parameter_sweeps.items():
            avg_sensitivity = np.mean(sweep.sensitivity_scores)
            sensitivity_ranking.append((param_name, avg_sensitivity))

        result.sensitivity_ranking = sorted(sensitivity_ranking, key=lambda x: x[1], reverse=True)

        return result

    def plot_sensitivity(
        self,
        sweep_result: ParameterSweep,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sensitivity analysis results.

        Args:
            sweep_result: Parameter sweep results
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Objective function vs parameter
        ax1.plot(sweep_result.parameter_values, sweep_result.objective_values, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel(f"{sweep_result.parameter_name}", fontsize=12)
        ax1.set_ylabel("Objective Value Z", fontsize=12)
        ax1.set_title(f"Objective Function vs {sweep_result.parameter_name}", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Sensitivity score
        ax2.plot(sweep_result.parameter_values, sweep_result.sensitivity_scores, 's-', color='orange', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='r', linestyle='--', label='High sensitivity threshold')
        ax2.set_xlabel(f"{sweep_result.parameter_name}", fontsize=12)
        ax2.set_ylabel("Sensitivity |S(θ)|", fontsize=12)
        ax2.set_title(f"Sensitivity Score vs {sweep_result.parameter_name}", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_sensitivity_heatmap(
        self,
        result: SensitivityResult,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sensitivity heatmap across all parameters.

        Args:
            result: Sensitivity analysis results
            show: Whether to display the plot
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        # Prepare data
        param_names = list(result.parameter_sweeps.keys())
        max_sensitivities = [
            max(result.parameter_sweeps[name].sensitivity_scores)
            for name in param_names
        ]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Reshape for heatmap (single row)
        data = np.array(max_sensitivities).reshape(1, -1)

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks([0])
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticklabels(['Sensitivity'])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Max Sensitivity |S(θ)|', rotation=270, labelpad=20)

        # Add text annotations
        for i, val in enumerate(max_sensitivities):
            text = ax.text(i, 0, f'{val:.2f}', ha='center', va='center',
                          color='black' if val < 0.5 else 'white', fontsize=12, fontweight='bold')

        ax.set_title('Parameter Sensitivity Heatmap', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()
