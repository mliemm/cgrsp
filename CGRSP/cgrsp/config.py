"""
CGRSP Configuration Module

Defines resource types, fleet configurations, temporal parameters,
and availability parameters for the GPU rendering scheduling simulator.
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class ResourceType(str, Enum):
    """
    GPU resource types available in the CGRSP fleet.

    **CGRSP supports exactly 4 GPU types:**
    - RTX_A5000: Fastest on average (24GB VRAM, 1.00x speed)
    - RTX_3090: 2nd (24GB VRAM, 1.00x speed)
    - RTX_A4500: 3rd (20GB VRAM, 0.84x speed)
    - RTX_A4000: Slowest on average (16GB VRAM, 0.74x speed)

    All empirical data and pricing models are based on these 4 GPU types
    from 6,234 real rendering jobs. No other GPU types are supported.
    """
    RTX_3090 = "RTX_3090"
    RTX_A5000 = "RTX_A5000"
    RTX_A4500 = "RTX_A4500"
    RTX_A4000 = "RTX_A4000"

    @classmethod
    def supported_gpus(cls) -> list:
        """Return list of supported GPU types."""
        return [cls.RTX_3090, cls.RTX_A5000, cls.RTX_A4500, cls.RTX_A4000]


@dataclass
class ResourceSpec:
    """Specifications for a GPU resource type."""
    name: str
    vram: int  # GB
    cuda_cores: int
    base_speed: float  # Relative rendering speed multiplier
    hourly_cost: float  # USD per hour


# Resource specifications based on NVIDIA GPU specs
RESOURCE_SPECS: Dict[ResourceType, ResourceSpec] = {
    ResourceType.RTX_3090: ResourceSpec(
        name="RTX 3090",
        vram=24,
        cuda_cores=10496,
        base_speed=1.00,  # empirical, A5000=1.00 baseline (see execution_times.py)
        hourly_cost=0.46,
    ),
    ResourceType.RTX_A5000: ResourceSpec(
        name="RTX A5000",
        vram=24,
        cuda_cores=8192,
        base_speed=1.00,  # fastest on average (baseline)
        hourly_cost=0.27,
    ),
    ResourceType.RTX_A4500: ResourceSpec(
        name="RTX A4500",
        vram=20,
        cuda_cores=7168,
        base_speed=0.84,
        hourly_cost=0.25, # RunPod Secure Price, 11 March 2026 snapshot (Taiwan time)
    ),
    ResourceType.RTX_A4000: ResourceSpec(
        name="RTX A4000",
        vram=16,
        cuda_cores=6144,
        base_speed=0.74,
        hourly_cost=0.25, # RunPod Secure Price 2026
    ),
}


@dataclass
class FleetConfig:
    """Configuration for a GPU fleet."""
    resource_type: ResourceType
    count: int


# Example fleet configurations
EXAMPLE_FLEETS: Dict[str, List[FleetConfig]] = {
    "small": [
        FleetConfig(ResourceType.RTX_3090, 4),
        FleetConfig(ResourceType.RTX_A4000, 8),
    ],
    "medium": [
        FleetConfig(ResourceType.RTX_3090, 8),
        FleetConfig(ResourceType.RTX_A5000, 4),
        FleetConfig(ResourceType.RTX_A4500, 8),
        FleetConfig(ResourceType.RTX_A4000, 16),
    ],
    "large": [
        FleetConfig(ResourceType.RTX_3090, 16),
        FleetConfig(ResourceType.RTX_A5000, 8),
        FleetConfig(ResourceType.RTX_A4500, 16),
        FleetConfig(ResourceType.RTX_A4000, 32),
    ],
}


@dataclass
class TemporalParameters:
    """Temporal parameters for the scheduling problem."""
    T_deadline: float = 3600.0   # 1 hour tight deadline (20% of jobs)
    simulation_horizon: float = 86400.0  # 24 hours in seconds
    time_step: float = 1.0  # Minimum time resolution (seconds)


@dataclass
class AvailabilityParameters:
    """
    Markov process parameters for node availability transitions.

    Simplified customer-only model:
    - λ_customer: Customer usage initiation rate (time-dependent)
    - μ_release: Customer usage release rate (time-dependent)

    This model focuses on customer usage patterns only.
    Maintenance and hardware failures are handled by the cloud provider
    and are not explicitly modeled.
    """
    # Transition TO Unavailable (from Idle or Running)
    lambda_customer_base: float = 0.1  # Base rate (per hour)

    # Customer usage patterns
    customer_usage_peak_multiplier: float = 2.0  # Peak hours multiplier
    customer_usage_peak_start: int = 9  # 9 AM
    customer_usage_peak_end: int = 17  # 5 PM
    customer_usage_off_peak_multiplier: float = 0.5  # Off-peak multiplier

    # Release rate parameters (per hour)
    mu_release_base: float = 0.2  # Base release rate (per hour)

    def get_customer_rate(self, hour: int) -> float:
        """
        Get customer usage initiation rate for a given hour.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Transition rate λ_customer(t) in per-second units
        """
        if self.customer_usage_peak_start <= hour <= self.customer_usage_peak_end:
            multiplier = self.customer_usage_peak_multiplier
        else:
            multiplier = self.customer_usage_off_peak_multiplier

        rate_per_hour = self.lambda_customer_base * multiplier
        return rate_per_hour / 3600.0  # Convert to per-second

    def get_release_rate(self, hour: int) -> float:
        """
        Get customer usage release rate for a given hour.

        Simplified model: release rate inversely proportional to initiation rate.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Transition rate μ_release(t) in per-second units
        """
        # Higher release rate outside business hours
        if self.customer_usage_peak_start <= hour <= self.customer_usage_peak_end:
            rate_per_hour = self.mu_release_base * 0.8  # Lower during business hours
        else:
            rate_per_hour = self.mu_release_base * 1.2  # Higher outside business hours

        return rate_per_hour / 3600.0


# Default configuration instances
DEFAULT_TEMPORAL = TemporalParameters()
DEFAULT_AVAILABILITY = AvailabilityParameters()

# Get total nodes in a fleet configuration
def get_fleet_size(fleet_configs: List[FleetConfig]) -> int:
    """Calculate total number of nodes in a fleet configuration."""
    return sum(config.count for config in fleet_configs)


# Get all resource types in a fleet
def get_fleet_resources(fleet_configs: List[FleetConfig]) -> List[ResourceType]:
    """Get list of resource types in a fleet configuration."""
    return list(set(config.resource_type for config in fleet_configs))
