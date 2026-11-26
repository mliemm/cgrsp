"""
CGRSP Failure Scenarios
Models node failures, maintenance windows, and correlated failures.

Failure Types:
  - Random Node Failures: Independent hardware failures with exponential inter-arrival
  - Correlated Failures: Failures affecting entire GPU types or racks
  - Scheduled Maintenance: Planned downtime windows
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta

from .execution_times import GPUType


class FailureType(Enum):
    """Types of failure scenarios."""
    RANDOM = "random"  # Independent random failures
    CORRELATED = "correlated"  # Failures affecting multiple nodes
    MAINTENANCE = "maintenance"  # Scheduled maintenance
    NONE = "none"  # No failures


@dataclass
class FailureEvent:
    """
    Represents a single failure event.

    Attributes:
        node_ids: Affected node IDs
        start_time: Failure start time
        duration: Failure duration (seconds)
        failure_type: Type of failure
        gpu_type: GPU type (for correlated failures)
        description: Human-readable description
    """
    node_ids: List[int]
    start_time: float
    duration: float
    failure_type: FailureType
    gpu_type: Optional[GPUType] = None
    description: str = ""

    @property
    def end_time(self) -> float:
        """Get failure end time."""
        return self.start_time + self.duration

    def affects_node(self, node_id: int) -> bool:
        """Check if failure affects a specific node."""
        return node_id in self.node_ids

    def is_active_at(self, time: float) -> bool:
        """Check if failure is active at a given time."""
        return self.start_time <= time < self.end_time


@dataclass
class FailureScenarioConfig:
    """Configuration for failure scenario generation."""
    failure_type: FailureType
    lambda_failure: float = 0.0001  # Failure rate (per second)
    mttr_hours: float = 10.0  # Mean time to repair (hours)
    correlation_factor: float = 0.1  # Probability of correlated failure
    maintenance_interval_hours: float = 168.0  # Weekly maintenance
    maintenance_duration_hours: float = 2.0  # Maintenance window duration
    affected_gpu_types: Optional[List[GPUType]] = None  # For correlated failures
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.lambda_failure < 0:
            raise ValueError(f"lambda_failure must be >= 0, got {self.lambda_failure}")

        if self.mttr_hours <= 0:
            raise ValueError(f"mttr_hours must be > 0, got {self.mttr_hours}")

        if not 0 <= self.correlation_factor <= 1:
            raise ValueError(
                f"correlation_factor must be in [0, 1], got {self.correlation_factor}"
            )

        if self.maintenance_interval_hours <= 0:
            raise ValueError(
                f"maintenance_interval_hours must be > 0, "
                f"got {self.maintenance_interval_hours}"
            )

        if self.maintenance_duration_hours <= 0:
            raise ValueError(
                f"maintenance_duration_hours must be > 0, "
                f"got {self.maintenance_duration_hours}"
            )


class FailureScenario:
    """
    Base class for failure scenario generators.
    """

    def __init__(self, config: FailureScenarioConfig):
        """
        Initialize failure scenario.

        Args:
            config: Failure scenario configuration
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)

        self.failure_events: List[FailureEvent] = []

    def generate(self, num_nodes: int, simulation_duration: float) -> List[FailureEvent]:
        """
        Generate failure events for a simulation.

        Args:
            num_nodes: Total number of nodes in the fleet
            simulation_duration: Total simulation time (seconds)

        Returns:
            List of failure events
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def get_failures_at_time(self, time: float) -> List[FailureEvent]:
        """Get all failure events active at a given time."""
        return [f for f in self.failure_events if f.is_active_at(time)]

    def get_affected_nodes_at_time(self, time: float) -> Set[int]:
        """Get set of node IDs affected by failures at a given time."""
        affected = set()
        for failure in self.get_failures_at_time(time):
            affected.update(failure.node_ids)
        return affected


class RandomNodeFailures(FailureScenario):
    """
    Models independent random node failures.

    Failures follow exponential distribution with rate λ_failure.
    Repair times follow exponential distribution with rate μ_repair.
    """

    def __init__(self, config: Optional[FailureScenarioConfig] = None):
        """
        Initialize random failure scenario.

        Args:
            config: Failure configuration. If None, uses defaults.
        """
        if config is None:
            config = FailureScenarioConfig(
                failure_type=FailureType.RANDOM,
                lambda_failure=0.0001,  # MTBF = 10000 hours
                mttr_hours=10.0
            )

        super().__init__(config)

        # Calculate repair rate from MTTR
        mttr_seconds = config.mttr_hours * 3600
        self.mu_repair = 1.0 / mttr_seconds

    def generate(self, num_nodes: int, simulation_duration: float) -> List[FailureEvent]:
        """
        Generate random failure events for all nodes.

        Args:
            num_nodes: Total number of nodes in the fleet
            simulation_duration: Total simulation time (seconds)

        Returns:
            List of failure events
        """
        self.failure_events = []

        for node_id in range(num_nodes):
            current_time = 0.0

            # Generate failures for this node
            while current_time < simulation_duration:
                # Sample time to next failure
                time_to_failure = np.random.exponential(1.0 / self.config.lambda_failure)
                current_time += time_to_failure

                if current_time >= simulation_duration:
                    break

                # Sample repair time
                repair_duration = np.random.exponential(1.0 / self.mu_repair)

                # Create failure event
                event = FailureEvent(
                    node_ids=[node_id],
                    start_time=current_time,
                    duration=repair_duration,
                    failure_type=FailureType.RANDOM,
                    description=f"Random failure on node {node_id}"
                )

                self.failure_events.append(event)

                # Move past repair time
                current_time += repair_duration

        return self.failure_events


class CorrelatedFailures(FailureScenario):
    """
    Models correlated failures affecting multiple nodes.

    Examples:
      - Rack-level power failures
      - GPU type-specific driver issues
      - Network switch failures
    """

    def __init__(self, config: FailureScenarioConfig):
        """
        Initialize correlated failure scenario.

        Args:
            config: Failure configuration
        """
        super().__init__(config)

        if config.affected_gpu_types is None:
            self.affected_gpu_types = list(GPUType)
        else:
            self.affected_gpu_types = config.affected_gpu_types

    def generate(
        self,
        num_nodes: int,
        simulation_duration: float,
        node_gpu_types: Optional[Dict[int, GPUType]] = None
    ) -> List[FailureEvent]:
        """
        Generate correlated failure events.

        Args:
            num_nodes: Total number of nodes in the fleet
            simulation_duration: Total simulation time (seconds)
            node_gpu_types: Mapping from node ID to GPU type (required)

        Returns:
            List of failure events

        Raises:
            ValueError: If node_gpu_types is not provided
        """
        if node_gpu_types is None:
            raise ValueError(
                "node_gpu_types must be provided for correlated failures"
            )

        self.failure_events = []
        current_time = 0.0

        # Group nodes by GPU type
        nodes_by_type = {}
        for node_id, gpu_type in node_gpu_types.items():
            if gpu_type not in nodes_by_type:
                nodes_by_type[gpu_type] = []
            nodes_by_type[gpu_type].append(node_id)

        # Generate correlated failures
        while current_time < simulation_duration:
            # Sample time to next correlated failure
            time_to_failure = np.random.exponential(
                1.0 / (self.config.lambda_failure * self.config.correlation_factor)
            )
            current_time += time_to_failure

            if current_time >= simulation_duration:
                break

            # Select GPU type to fail
            affected_type = np.random.choice(self.affected_gpu_types)

            # Sample repair time (longer for correlated failures)
            repair_duration = np.random.exponential(
                self.config.mttr_hours * 3600 * 2
            )

            # Get affected nodes
            affected_nodes = nodes_by_type.get(affected_type, [])

            if not affected_nodes:
                continue

            # Create failure event
            event = FailureEvent(
                node_ids=affected_nodes.copy(),
                start_time=current_time,
                duration=repair_duration,
                failure_type=FailureType.CORRELATED,
                gpu_type=affected_type,
                description=f"Correlated failure affecting all {affected_type.value} nodes"
            )

            self.failure_events.append(event)

            # Move past repair time
            current_time += repair_duration

        return self.failure_events


class ScheduledMaintenance(FailureScenario):
    """
    Models scheduled maintenance windows.

    Maintenance occurs at regular intervals and affects a subset of nodes.
    """

    def __init__(self, config: Optional[FailureScenarioConfig] = None):
        """
        Initialize scheduled maintenance scenario.

        Args:
            config: Failure configuration. If None, uses defaults.
        """
        if config is None:
            config = FailureScenarioConfig(
                failure_type=FailureType.MAINTENANCE,
                maintenance_interval_hours=168.0,  # Weekly
                maintenance_duration_hours=2.0
            )

        super().__init__(config)

    def generate(
        self,
        num_nodes: int,
        simulation_duration: float,
        nodes_per_window: int = 4
    ) -> List[FailureEvent]:
        """
        Generate scheduled maintenance events.

        Args:
            num_nodes: Total number of nodes in the fleet
            simulation_duration: Total simulation time (seconds)
            nodes_per_window: Number of nodes affected per maintenance window

        Returns:
            List of failure events
        """
        self.failure_events = []

        # Calculate maintenance interval and duration
        interval_seconds = self.config.maintenance_interval_hours * 3600
        duration_seconds = self.config.maintenance_duration_hours * 3600

        # Generate maintenance windows
        current_time = 0.0
        node_idx = 0

        while current_time < simulation_duration:
            # Select nodes for this maintenance window
            start_node = node_idx % num_nodes
            end_node = min(start_node + nodes_per_window, num_nodes)

            affected_nodes = list(range(start_node, end_node))
            node_idx = end_node

            # Create maintenance event
            event = FailureEvent(
                node_ids=affected_nodes,
                start_time=current_time,
                duration=duration_seconds,
                failure_type=FailureType.MAINTENANCE,
                description=f"Scheduled maintenance for nodes {affected_nodes}"
            )

            self.failure_events.append(event)

            # Move to next maintenance window
            current_time += interval_seconds

        return self.failure_events

    def generate_maintenance_schedule(
        self,
        num_nodes: int,
        simulation_duration: float,
        start_date: Optional[datetime] = None
    ) -> List[Tuple[datetime, datetime, List[int]]]:
        """
        Generate maintenance schedule with real timestamps.

        Args:
            num_nodes: Total number of nodes in the fleet
            simulation_duration: Total simulation time (seconds)
            start_date: Start date for simulation (default: now)

        Returns:
            List of (start_time, end_time, affected_nodes) tuples
        """
        if start_date is None:
            start_date = datetime.now()

        schedule = []
        current_time = start_date
        interval = timedelta(hours=self.config.maintenance_interval_hours)
        duration = timedelta(hours=self.config.maintenance_duration_hours)

        sim_end_date = start_date + timedelta(seconds=simulation_duration)
        node_idx = 0

        while current_time < sim_end_date:
            # Select nodes for this window
            start_node = node_idx % num_nodes
            end_node = min(start_node + 4, num_nodes)  # 4 nodes per window
            affected_nodes = list(range(start_node, end_node))
            node_idx = end_node

            end_time = current_time + duration

            schedule.append((current_time, end_time, affected_nodes))

            # Move to next window
            current_time += interval

        return schedule


class FailureScenarioBuilder:
    """
    Builder for creating combined failure scenarios.
    """

    @staticmethod
    def no_failures() -> FailureScenario:
        """Create a scenario with no failures."""
        config = FailureScenarioConfig(failure_type=FailureType.NONE)
        return FailureScenario(config)

    @staticmethod
    def random_failures(
        mtbf_hours: float = 10000.0,
        mttr_hours: float = 10.0,
        seed: Optional[int] = None
    ) -> RandomNodeFailures:
        """
        Create random failure scenario.

        Args:
            mtbf_hours: Mean time between failures (hours)
            mttr_hours: Mean time to repair (hours)
            seed: Random seed

        Returns:
            Random failure scenario
        """
        config = FailureScenarioConfig(
            failure_type=FailureType.RANDOM,
            lambda_failure=1.0 / (mtbf_hours * 3600),
            mttr_hours=mttr_hours,
            seed=seed
        )

        return RandomNodeFailures(config)

    @staticmethod
    def correlated_failures(
        mtbf_hours: float = 10000.0,
        mttr_hours: float = 10.0,
        correlation_factor: float = 0.1,
        affected_gpu_types: Optional[List[GPUType]] = None,
        seed: Optional[int] = None
    ) -> CorrelatedFailures:
        """
        Create correlated failure scenario.

        Args:
            mtbf_hours: Mean time between failures (hours)
            mttr_hours: Mean time to repair (hours)
            correlation_factor: Probability of correlated vs independent failure
            affected_gpu_types: GPU types affected by correlated failures
            seed: Random seed

        Returns:
            Correlated failure scenario
        """
        config = FailureScenarioConfig(
            failure_type=FailureType.CORRELATED,
            lambda_failure=1.0 / (mtbf_hours * 3600),
            mttr_hours=mttr_hours,
            correlation_factor=correlation_factor,
            affected_gpu_types=affected_gpu_types,
            seed=seed
        )

        return CorrelatedFailures(config)

    @staticmethod
    def scheduled_maintenance(
        interval_hours: float = 168.0,  # Weekly
        duration_hours: float = 2.0,
        seed: Optional[int] = None
    ) -> ScheduledMaintenance:
        """
        Create scheduled maintenance scenario.

        Args:
            interval_hours: Time between maintenance windows (hours)
            duration_hours: Duration of each maintenance window (hours)
            seed: Random seed

        Returns:
            Scheduled maintenance scenario
        """
        config = FailureScenarioConfig(
            failure_type=FailureType.MAINTENANCE,
            maintenance_interval_hours=interval_hours,
            maintenance_duration_hours=duration_hours,
            seed=seed
        )

        return ScheduledMaintenance(config)


if __name__ == "__main__":
    # Example usage
    print("CGRSP Failure Scenarios")
    print("=" * 50)

    # Test random failures
    print("\nRandom Failures:")
    print("-" * 40)

    random_scenario = FailureScenarioBuilder.random_failures(
        mtbf_hours=1000,  # Faster failures for demo
        mttr_hours=2,
        seed=42
    )

    events = random_scenario.generate(num_nodes=10, simulation_duration=3600 * 24)  # 24 hours

    print(f"Generated {len(events)} failure events")
    print(f"First 5 events:")
    for event in events[:5]:
        print(f"  Node {event.node_ids[0]}: "
              f"start={event.start_time/3600:.1f}h, "
              f"duration={event.duration/3600:.2f}h")

    # Test scheduled maintenance
    print("\n\nScheduled Maintenance:")
    print("-" * 40)

    maint_scenario = FailureScenarioBuilder.scheduled_maintenance(
        interval_hours=12,  # Every 12 hours for demo
        duration_hours=1
    )

    events = maint_scenario.generate(
        num_nodes=10,
        simulation_duration=3600 * 48,  # 48 hours
        nodes_per_window=2
    )

    print(f"Generated {len(events)} maintenance windows")
    print(f"First 3 windows:")
    for event in events[:3]:
        print(f"  Nodes {event.node_ids}: "
              f"start={event.start_time/3600:.1f}h, "
              f"duration={event.duration/3600:.1f}h")
