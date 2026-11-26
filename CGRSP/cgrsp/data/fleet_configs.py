"""
CGRSP Fleet Configurations
Predefined GPU fleet configurations for different use cases and scales.

Fleet Profiles:
  - Balanced: Equal distribution of all GPU types
  - High-Performance Skewed: Emphasis on RTX 3090
  - Cost-Optimized: Emphasis on affordable GPUs (A4500, A4000)
  - Small Scale: Testing configuration (8 nodes)
  - Enterprise Scale: Production configuration (100 nodes)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

from .execution_times import GPUType, GPU_SPECIFICATIONS, PRICING_LOOKUP


class FleetProfile(Enum):
    """Predefined fleet configuration profiles."""
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    COST_OPTIMIZED = "cost_optimized"
    SMALL_SCALE = "small_scale"
    ENTERPRISE_SCALE = "enterprise_scale"


@dataclass
class FleetConfiguration:
    """
    Configuration for a heterogeneous GPU fleet.

    Attributes:
        name: Configuration name
        description: Human-readable description
        gpu_counts: Dictionary mapping GPUType to count
        availability_params: Stochastic availability parameters
    """
    name: str
    description: str
    gpu_counts: Dict[GPUType, int] = field(default_factory=dict)
    availability_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.gpu_counts:
            raise ValueError("gpu_counts cannot be empty")

        if any(count < 0 for count in self.gpu_counts.values()):
            raise ValueError("GPU counts must be non-negative")

    @property
    def total_nodes(self) -> int:
        """Get total number of nodes in the fleet."""
        return sum(self.gpu_counts.values())

    @property
    def total_cuda_cores(self) -> int:
        """Get total CUDA cores across all nodes."""
        return sum(
            count * GPU_SPECIFICATIONS[gpu_type]["cuda_cores"]
            for gpu_type, count in self.gpu_counts.items()
        )

    @property
    def total_cost_per_second(self) -> float:
        """Get total operational cost per second (when all nodes running)."""
        return sum(
            count * PRICING_LOOKUP[gpu_type]
            for gpu_type, count in self.gpu_counts.items()
        )

    @property
    def avg_cost_per_node(self) -> float:
        """Get average cost per node per second."""
        if self.total_nodes == 0:
            return 0.0
        return self.total_cost_per_second / self.total_nodes

    @property
    def total_capacity(self) -> float:
        """Get total normalized capacity across all nodes."""
        return sum(
            count * GPU_SPECIFICATIONS[gpu_type]["relative_performance"]
            for gpu_type, count in self.gpu_counts.items()
        )

    def get_gpu_distribution(self) -> Dict[str, int]:
        """Get GPU distribution as string dictionary."""
        return {
            gpu_type.value: count
            for gpu_type, count in self.gpu_counts.items()
        }

    def get_nodes_by_type(self, gpu_type: GPUType) -> int:
        """Get number of nodes of a specific GPU type."""
        return self.gpu_counts.get(gpu_type, 0)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "total_nodes": self.total_nodes,
            "gpu_distribution": self.get_gpu_distribution(),
            "total_cuda_cores": self.total_cuda_cores,
            "total_cost_per_second": self.total_cost_per_second,
            "avg_cost_per_node": self.avg_cost_per_node,
            "total_capacity": self.total_capacity,
            "availability_params": self.availability_params,
        }


# Default availability parameters for stochastic model
DEFAULT_AVAILABILITY_PARAMS = {
    "lambda_customer": 0.02,      # Customer demand rate
    "lambda_maintenance": 0.005,   # Maintenance event rate
    "lambda_failure": 0.0001,      # Hardware failure rate (MTBF = 10000 hours)
    "mu_release": 0.015,          # Resource release rate
    "mu_repair": 0.1,             # Repair rate (MTTR = 10 hours)
}

# Predefined fleet configurations

# Balanced Fleet: Equal distribution of all GPU types
BALANCED_FLEET = FleetConfiguration(
    name="Balanced Fleet",
    description="Equal distribution of all GPU types (N=24). Balanced performance and cost.",
    gpu_counts={
        GPUType.RTX_3090: 6,
        GPUType.RTX_A5000: 6,
        GPUType.RTX_A4500: 6,
        GPUType.RTX_A4000: 6,
    },
    availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
)

# High-Performance Fleet: Skewed toward fastest GPUs
HIGH_PERFORMANCE_FLEET = FleetConfiguration(
    name="High-Performance Fleet",
    description="Skewed toward RTX 3090 for maximum throughput (N=20).",
    gpu_counts={
        GPUType.RTX_3090: 12,
        GPUType.RTX_A5000: 4,
        GPUType.RTX_A4500: 2,
        GPUType.RTX_A4000: 2,
    },
    availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
)

# Cost-Optimized Fleet: Emphasis on affordable GPUs
COST_OPTIMIZED_FLEET = FleetConfiguration(
    name="Cost-Optimized Fleet",
    description="Emphasis on affordable GPUs (A4500, A4000) for lower cost (N=24).",
    gpu_counts={
        GPUType.RTX_3090: 4,
        GPUType.RTX_A5000: 2,
        GPUType.RTX_A4500: 10,
        GPUType.RTX_A4000: 8,
    },
    availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
)

# Small Scale: Testing configuration
SMALL_SCALE_FLEET = FleetConfiguration(
    name="Small Scale Fleet",
    description="Small-scale configuration for testing and development (N=8).",
    gpu_counts={
        GPUType.RTX_3090: 2,
        GPUType.RTX_A5000: 2,
        GPUType.RTX_A4500: 2,
        GPUType.RTX_A4000: 2,
    },
    availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
)

# Enterprise Scale: Production configuration
ENTERPRISE_SCALE_FLEET = FleetConfiguration(
    name="Enterprise Scale Fleet",
    description="Large-scale production fleet (N=100).",
    gpu_counts={
        GPUType.RTX_3090: 25,
        GPUType.RTX_A5000: 25,
        GPUType.RTX_A4500: 25,
        GPUType.RTX_A4000: 25,
    },
    availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
)


class FleetConfigurationBuilder:
    """
    Builder for creating custom fleet configurations.
    """

    @staticmethod
    def balanced(num_per_type: int = 6) -> FleetConfiguration:
        """
        Create a balanced fleet with equal GPU distribution.

        Args:
            num_per_type: Number of nodes per GPU type

        Returns:
            Fleet configuration
        """
        return FleetConfiguration(
            name=f"Balanced Fleet ({num_per_type} per type)",
            description=f"Equal distribution: {num_per_type} nodes per GPU type",
            gpu_counts={
                GPUType.RTX_3090: num_per_type,
                GPUType.RTX_A5000: num_per_type,
                GPUType.RTX_A4500: num_per_type,
                GPUType.RTX_A4000: num_per_type,
            },
            availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
        )

    @staticmethod
    def performance_skewed(
        rtx_3090_count: int = 12,
        other_count: int = 4
    ) -> FleetConfiguration:
        """
        Create a performance-skewed fleet.

        Args:
            rtx_3090_count: Number of RTX 3090 nodes
            other_count: Number of nodes for other GPU types

        Returns:
            Fleet configuration
        """
        return FleetConfiguration(
            name=f"Performance Skewed Fleet",
            description=f"High-performance skew: {rtx_3090_count}x RTX 3090, {other_count}x others",
            gpu_counts={
                GPUType.RTX_3090: rtx_3090_count,
                GPUType.RTX_A5000: other_count,
                GPUType.RTX_A4500: other_count,
                GPUType.RTX_A4000: other_count,
            },
            availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
        )

    @staticmethod
    def cost_optimized(
        affordable_count: int = 10,
        premium_count: int = 4
    ) -> FleetConfiguration:
        """
        Create a cost-optimized fleet.

        Args:
            affordable_count: Number of affordable GPU nodes (A4500, A4000)
            premium_count: Number of premium GPU nodes (RTX 3090, A5000)

        Returns:
            Fleet configuration
        """
        return FleetConfiguration(
            name=f"Cost-Optimized Fleet",
            description=f"Cost optimization: {affordable_count}x affordable, {premium_count}x premium",
            gpu_counts={
                GPUType.RTX_3090: premium_count,
                GPUType.RTX_A5000: premium_count,
                GPUType.RTX_A4500: affordable_count,
                GPUType.RTX_A4000: affordable_count,
            },
            availability_params=DEFAULT_AVAILABILITY_PARAMS.copy()
        )

    @staticmethod
    def custom(
        rtx_3090: int,
        rtx_a5000: int,
        rtx_a4500: int,
        rtx_a4000: int,
        **availability_params
    ) -> FleetConfiguration:
        """
        Create a custom fleet configuration.

        Args:
            rtx_3090: Number of RTX 3090 nodes
            rtx_a5000: Number of RTX A5000 nodes
            rtx_a4500: Number of RTX A4500 nodes
            rtx_a4000: Number of RTX A4000 nodes
            **availability_params: Optional availability parameters

        Returns:
            Fleet configuration
        """
        params = DEFAULT_AVAILABILITY_PARAMS.copy()
        params.update(availability_params)

        total_nodes = rtx_3090 + rtx_a5000 + rtx_a4500 + rtx_a4000

        return FleetConfiguration(
            name=f"Custom Fleet ({total_nodes} nodes)",
            description=f"Custom configuration: {rtx_3090}x 3090, {rtx_a5000}x A5000, "
                       f"{rtx_a4500}x A4500, {rtx_a4000}x A4000",
            gpu_counts={
                GPUType.RTX_3090: rtx_3090,
                GPUType.RTX_A5000: rtx_a5000,
                GPUType.RTX_A4500: rtx_a4500,
                GPUType.RTX_A4000: rtx_a4000,
            },
            availability_params=params
        )

    @staticmethod
    def from_profile(profile: FleetProfile) -> FleetConfiguration:
        """
        Get fleet configuration from predefined profile.

        Args:
            profile: Fleet profile enum

        Returns:
            Fleet configuration

        Raises:
            ValueError: If profile is not recognized
        """
        fleets = {
            FleetProfile.BALANCED: BALANCED_FLEET,
            FleetProfile.HIGH_PERFORMANCE: HIGH_PERFORMANCE_FLEET,
            FleetProfile.COST_OPTIMIZED: COST_OPTIMIZED_FLEET,
            FleetProfile.SMALL_SCALE: SMALL_SCALE_FLEET,
            FleetProfile.ENTERPRISE_SCALE: ENTERPRISE_SCALE_FLEET,
        }

        if profile not in fleets:
            raise ValueError(
                f"Unknown fleet profile: {profile}. "
                f"Must be one of {list(fleets.keys())}"
            )

        # Return a copy to avoid modifying original
        config = fleets[profile]
        return FleetConfiguration(
            name=config.name,
            description=config.description,
            gpu_counts=config.gpu_counts.copy(),
            availability_params=config.availability_params.copy()
        )


def get_fleet_config(profile: FleetProfile) -> FleetConfiguration:
    """
    Get fleet configuration for a predefined profile.

    Args:
        profile: Fleet profile enum

    Returns:
        Fleet configuration
    """
    return FleetConfigurationBuilder.from_profile(profile)


if __name__ == "__main__":
    # Example usage
    print("CGRSP Fleet Configurations")
    print("=" * 50)

    # Display all predefined configurations
    profiles = [
        FleetProfile.BALANCED,
        FleetProfile.HIGH_PERFORMANCE,
        FleetProfile.COST_OPTIMIZED,
        FleetProfile.SMALL_SCALE,
        FleetProfile.ENTERPRISE_SCALE,
    ]

    for profile in profiles:
        config = get_fleet_config(profile)
        print(f"\n{profile.value.upper()}:")
        print("-" * 40)
        print(f"  Description: {config.description}")
        print(f"  Total nodes: {config.total_nodes}")
        print(f"  GPU distribution: {config.get_gpu_distribution()}")
        print(f"  Total CUDA cores: {config.total_cuda_cores:,}")
        print(f"  Cost per second: ${config.total_cost_per_second:.4f}")
        print(f"  Avg cost per node: ${config.avg_cost_per_node:.6f}")
        print(f"  Total capacity: {config.total_capacity:.2f}")

    # Example of building custom fleet
    print("\n\nCUSTOM FLEET EXAMPLE:")
    print("-" * 40)

    custom = FleetConfigurationBuilder.custom(
        rtx_3090=8,
        rtx_a5000=4,
        rtx_a4500=12,
        rtx_a4000=8
    )

    print(f"Name: {custom.name}")
    print(f"Description: {custom.description}")
    print(f"Total nodes: {custom.total_nodes}")
    print(f"GPU distribution: {custom.get_gpu_distribution()}")
