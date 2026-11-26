"""
CGRSP Data Package
Provides empirical data models and generators for Cloud GPU Rendering scheduling.
"""

from .execution_times import (
    ExecutionTimeModel,
    GPUType,
    COMPLEXITY_LEVELS,
    EXECUTION_TIME_LOOKUP,
    PRICING_LOOKUP,
    get_execution_time,
    get_pricing
)

from .job_generator import (
    JobGenerator,
    JobGeneratorConfig,
    COMPLEXITY_DISTRIBUTION,
    DEFAULT_DEADLINE_DURATION
)

from .scenarios import (
    WorkloadScenario,
    ScenarioType,
    get_scenario_config,
    LIGHT_LOAD_CONFIG,
    MODERATE_LOAD_CONFIG,
    HEAVY_LOAD_CONFIG,
    PEAK_HOUR_CONFIG
)

from .fleet_configs import (
    FleetConfiguration,
    FleetProfile,
    BALANCED_FLEET,
    HIGH_PERFORMANCE_FLEET,
    COST_OPTIMIZED_FLEET,
    SMALL_SCALE_FLEET,
    ENTERPRISE_SCALE_FLEET
)

from .failure_scenarios import (
    FailureScenario,
    FailureType,
    RandomNodeFailures,
    CorrelatedFailures,
    ScheduledMaintenance
)

from .availability import (
    AvailabilityModel,
    AvailabilityConfig,
    DEFAULT_AVAILABILITY_CONFIG,
    TimeVaryingAvailability,
    get_customer_demand_rate,
    get_release_rate
)

__all__ = [
    # Execution times
    "ExecutionTimeModel",
    "GPUType",
    "COMPLEXITY_LEVELS",
    "EXECUTION_TIME_LOOKUP",
    "PRICING_LOOKUP",
    "get_execution_time",
    "get_pricing",

    # Job generator
    "JobGenerator",
    "JobGeneratorConfig",
    "COMPLEXITY_DISTRIBUTION",
    "DEFAULT_DEADLINE_DURATION",

    # Scenarios
    "WorkloadScenario",
    "ScenarioType",
    "get_scenario_config",
    "LIGHT_LOAD_CONFIG",
    "MODERATE_LOAD_CONFIG",
    "HEAVY_LOAD_CONFIG",
    "PEAK_HOUR_CONFIG",

    # Fleet configurations
    "FleetConfiguration",
    "FleetProfile",
    "BALANCED_FLEET",
    "HIGH_PERFORMANCE_FLEET",
    "COST_OPTIMIZED_FLEET",
    "SMALL_SCALE_FLEET",
    "ENTERPRISE_SCALE_FLEET",

    # Failure scenarios
    "FailureScenario",
    "FailureType",
    "RandomNodeFailures",
    "CorrelatedFailures",
    "ScheduledMaintenance",

    # Availability
    "AvailabilityModel",
    "AvailabilityConfig",
    "DEFAULT_AVAILABILITY_CONFIG",
    "TimeVaryingAvailability",
    "get_customer_demand_rate",
    "get_maintenance_rate",
    "get_release_rate",
]
