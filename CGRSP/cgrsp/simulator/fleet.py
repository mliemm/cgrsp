"""
GPU Fleet Manager Module

Implements the GPU fleet. The cloud experiments use the on-demand Secure Cloud
model (RunPodCloudFleetManager): a slot is only ever Idle or Running, a running
slot is never reclaimed, and market scarcity is realised as a stock-status
dependent provisioning delay at dispatch (see runpod_availability), not as a
node-failure transition. The generic Markov availability machinery (Unavailable
state, execute_stochastic_transition, release/repair rates) is retained for the
non-cloud fleet but is not exercised by the cloud experiments. The slot model
derives from the SPARS machine model (Amrizal et al., 2026).
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum

from cgrsp.config import ResourceType, RESOURCE_SPECS, AvailabilityParameters


class NodeState(str, Enum):
    """Node states in the fleet."""
    UNAVAILABLE = "unavailable"
    IDLE = "idle"
    RUNNING = "running"


class UnavailableReason(str, Enum):
    """Reasons for node unavailability."""
    CUSTOMER = "customer"  # External customer usage
    MAINTENANCE = "maintenance"  # Scheduled maintenance
    FAILURE = "failure"  # Hardware failure


@dataclass
class Node:
    """
    Represents a single GPU slot in the fleet.

    Under the on-demand Secure Cloud model used in the experiments a slot is
    only ever Idle or Running, with transitions:
    - Idle -> Running (job assignment)
    - Running -> Idle (job completion)
    A running slot is never reclaimed (no preemption); provisioning scarcity is
    realised as a start-time delay (see runpod_availability), not as a
    node-failure transition. The UNAVAILABLE state and the
    execute_stochastic_transition machinery are retained for the generic
    (non-cloud) fleet but are not exercised by the cloud experiments.
    """

    node_id: int
    resource_type: ResourceType
    state: NodeState = NodeState.IDLE
    current_job_id: Optional[int] = None
    unavailable_reason: Optional[UnavailableReason] = None
    state_entry_time: float = 0.0
    job_start_time: float = 0.0  # Track when the current job started (for cost calculation)
    custom_price_per_sec: Optional[float] = None # For cloud/external pricing

    def __post_init__(self):
        """Initialize node with resource specifications."""
        self.spec = RESOURCE_SPECS[self.resource_type]

    def get_speed_multiplier(self) -> float:
        """Get the rendering speed multiplier for this node."""
        return self.spec.base_speed

    def get_hourly_cost(self) -> float:
        """Get the hourly operational cost for this node."""
        if self.custom_price_per_sec is not None:
            return self.custom_price_per_sec * 3600.0
        return self.spec.hourly_cost

    def get_per_second_cost(self) -> float:
        """Get the per-second operational cost for this node."""
        if self.custom_price_per_sec is not None:
            return self.custom_price_per_sec

        from cgrsp.data.execution_times import ExecutionTimeModel, GPUType

        gpu_type_map = {
            ResourceType.RTX_3090: GPUType.RTX_3090,
            ResourceType.RTX_A5000: GPUType.RTX_A5000,
            ResourceType.RTX_A4500: GPUType.RTX_A4500,
            ResourceType.RTX_A4000: GPUType.RTX_A4000,
        }

        gpu_type = gpu_type_map.get(self.resource_type, GPUType.RTX_3090)
        return ExecutionTimeModel.get_pricing(gpu_type)

    def get_vram(self) -> int:
        """Get VRAM capacity in GB."""
        return self.spec.vram

    def can_accept_job(self, required_memory: Optional[float] = None) -> bool:
        """
        Check if node can accept a new job.

        Args:
            required_memory: Optional VRAM requirement in GB

        Returns:
            True if node is idle and has sufficient VRAM
        """
        if self.state != NodeState.IDLE:
            return False

        if required_memory is not None:
            return required_memory <= self.spec.vram

        return True

    def assign_job(self, job_id: int, current_time: float) -> bool:
        """
        Assign a job to this node.

        Args:
            job_id: ID of job to assign
            current_time: Current simulation time

        Returns:
            True if job was assigned successfully
        """
        if not self.can_accept_job():
            return False

        self.state = NodeState.RUNNING
        self.current_job_id = job_id
        self.state_entry_time = current_time
        self.job_start_time = current_time  # Track job start time for cost calculation
        return True

    def complete_job(self, current_time: float) -> Optional[int]:
        """
        Complete the current job and transition to idle.

        Args:
            current_time: Current simulation time

        Returns:
            Completed job ID, or None if no job was running
        """
        if self.state != NodeState.RUNNING:
            return None

        job_id = self.current_job_id
        self.state = NodeState.IDLE
        self.current_job_id = None
        self.state_entry_time = current_time
        # Don't reset job_start_time here - it's needed for cost calculation in _handle_job_completion
        return job_id

    def transition_to_unavailable(
        self,
        reason: UnavailableReason,
        current_time: float
    ) -> bool:
        """
        Transition node to unavailable state.

        Args:
            reason: Reason for unavailability
            current_time: Current simulation time

        Returns:
            True if transition was successful
        """
        if self.state == NodeState.UNAVAILABLE:
            return False

        # Note: In a real system, running jobs would be interrupted
        # For simplicity, we transition regardless of current state
        self.state = NodeState.UNAVAILABLE
        self.unavailable_reason = reason
        self.current_job_id = None
        self.state_entry_time = current_time
        return True

    def transition_to_idle(self, current_time: float) -> bool:
        """
        Transition node from unavailable to idle.

        Args:
            current_time: Current simulation time

        Returns:
            True if transition was successful
        """
        if self.state != NodeState.UNAVAILABLE:
            return False

        self.state = NodeState.IDLE
        self.unavailable_reason = None
        self.current_job_id = None
        self.state_entry_time = current_time
        return True

    def get_time_in_state(self, current_time: float) -> float:
        """
        Get time spent in current state.

        Args:
            current_time: Current simulation time

        Returns:
            Time in current state (seconds)
        """
        return current_time - self.state_entry_time


class FleetManager:
    """
    Manages the GPU fleet with stochastic availability dynamics.

    Implements a continuous-time Markov process where nodes transition
    between states based on time-dependent transition rates.
    """

    def __init__(
        self,
        fleet_config: List[tuple],
        availability_params: Optional[AvailabilityParameters] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the fleet manager.

        Args:
            fleet_config: List of (ResourceType, count) tuples
            availability_params: Availability parameters (uses defaults if None)
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        self.availability_params = availability_params or AvailabilityParameters()
        self.nodes: Dict[int, Node] = {}
        self._next_node_id = 0

        # Create nodes from configuration
        for resource_type, count in fleet_config:
            for _ in range(count):
                self._add_node(resource_type)

    def _add_node(self, resource_type: ResourceType) -> Node:
        """Add a new node to the fleet."""
        node = Node(node_id=self._next_node_id, resource_type=resource_type)
        self.nodes[self._next_node_id] = node
        self._next_node_id += 1
        return node

    def get_node(self, node_id: int) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the fleet."""
        return list(self.nodes.values())

    def get_nodes_by_state(self, state: NodeState) -> List[Node]:
        """Get all nodes in a specific state."""
        return [
            node for node in self.nodes.values()
            if node.state == state
        ]

    def get_nodes_by_resource_type(
        self,
        resource_type: ResourceType,
        state: Optional[NodeState] = None
    ) -> List[Node]:
        """
        Get nodes of a specific resource type.

        Args:
            resource_type: GPU resource type
            state: Optional state filter

        Returns:
            List of matching nodes
        """
        nodes = [
            node for node in self.nodes.values()
            if node.resource_type == resource_type
        ]

        if state is not None:
            nodes = [n for n in nodes if n.state == state]

        return nodes

    def get_idle_nodes(self) -> List[Node]:
        """Get all idle nodes."""
        return self.get_nodes_by_state(NodeState.IDLE)

    def get_available_nodes(
        self,
        required_memory: Optional[float] = None
    ) -> List[Node]:
        """
        Get nodes that can accept new jobs.

        Args:
            required_memory: Optional VRAM requirement

        Returns:
            List of available nodes
        """
        return [
            node for node in self.get_idle_nodes()
            if node.can_accept_job(required_memory)
        ]

    def get_fleet_size(self) -> int:
        """Get total number of nodes in the fleet."""
        return len(self.nodes)

    def get_fleet_utilization(self, current_time: float) -> float:
        """
        Calculate current fleet utilization.

        Args:
            current_time: Current simulation time

        Returns:
            Fraction of nodes in running state
        """
        total = len(self.nodes)
        if total == 0:
            return 0.0

        running = len(self.get_nodes_by_state(NodeState.RUNNING))
        return running / total

    def assign_job_to_node(
        self,
        node_id: int,
        job_id: int,
        current_time: float
    ) -> bool:
        """
        Assign a job to a specific node.

        Args:
            node_id: Target node ID
            job_id: Job to assign
            current_time: Current simulation time

        Returns:
            True if assignment was successful
        """
        node = self.get_node(node_id)
        if node is None:
            return False

        return node.assign_job(job_id, current_time)

    def complete_job_on_node(
        self,
        node_id: int,
        current_time: float
    ) -> Optional[int]:
        """
        Complete a job on a specific node.

        Args:
            node_id: Node ID
            current_time: Current simulation time

        Returns:
            Completed job ID, or None if no job was running
        """
        node = self.get_node(node_id)
        if node is None:
            return None

        return node.complete_job(current_time)

    def calculate_time_to_next_transition(
        self,
        node_id: int,
        current_time: float
    ) -> float:
        """
        Calculate time until next stochastic state transition.

        Uses the inverse CDF method to sample from exponential distribution
        based on current transition rates.

        Args:
            node_id: Node ID
            current_time: Current simulation time

        Returns:
            Time to next transition (seconds), or infinity if no transition possible
        """
        node = self.get_node(node_id)
        if node is None:
            return float('inf')

        # Get current hour for time-dependent rates
        hour = int(current_time / 3600) % 24

        # Calculate total transition rate based on current state
        total_rate = 0.0

        if node.state == NodeState.IDLE:
            # Idle can transition to: Unavailable (customer usage only)
            # Simplified customer-only availability model
            lambda_customer = self.availability_params.get_customer_rate(hour)
            total_rate = lambda_customer

        elif node.state == NodeState.RUNNING:
            # Running typically cannot transition to unavailable (job occupies the node)
            # In customer-only model, running jobs are protected from preemption
            total_rate = 0.0

        elif node.state == NodeState.UNAVAILABLE:
            # Unavailable transitions back to Idle (customer releases GPU)
            mu_release = self.availability_params.get_release_rate(hour)
            total_rate = mu_release

        # Sample time to transition from exponential distribution
        if total_rate > 0:
            return random.expovariate(total_rate)
        else:
            return float('inf')

    def sample_next_transition_type(
        self,
        node_id: int,
        current_time: float
    ) -> Optional[tuple[str, UnavailableReason]]:
        """
        Sample the type of next state transition.

        Args:
            node_id: Node ID
            current_time: Current simulation time

        Returns:
            Tuple of (new_state, reason) or None if no transition possible
        """
        node = self.get_node(node_id)
        if node is None:
            return None

        hour = int(current_time / 3600) % 24

        # Define possible transitions and their probabilities
        transitions = []

        if node.state == NodeState.IDLE:
            # Collect all possible transitions (customer-only model)
            lambda_customer = self.availability_params.get_customer_rate(hour)

            transitions = [
                (lambda_customer, UnavailableReason.CUSTOMER),
            ]

        elif node.state == NodeState.RUNNING:
            # Running nodes typically don't transition to unavailable in customer-only model
            transitions = [
                (0.0, UnavailableReason.CUSTOMER),  # No transitions while running
            ]

        elif node.state == NodeState.UNAVAILABLE:
            # Only transition is back to idle
            return ("idle", None)

        # Sample based on rates
        if not transitions:
            return None

        # Normalize rates to probabilities
        total_rate = sum(rate for rate, _ in transitions)
        if total_rate == 0:
            return None

        # Weighted random choice
        rates, reasons = zip(*transitions)
        chosen_reason = random.choices(reasons, weights=rates, k=1)[0]

        return ("unavailable", chosen_reason)

    def execute_stochastic_transition(
        self,
        node_id: int,
        current_time: float
    ) -> bool:
        """
        Execute a stochastic state transition on a node.

        Args:
            node_id: Node ID
            current_time: Current simulation time

        Returns:
            True if a transition occurred
        """
        node = self.get_node(node_id)
        if node is None:
            return False

        transition = self.sample_next_transition_type(node_id, current_time)
        if transition is None:
            return False

        new_state, reason = transition

        if new_state == "unavailable":
            return node.transition_to_unavailable(reason, current_time)
        elif new_state == "idle":
            return node.transition_to_idle(current_time)

        return False

    def get_state_summary(self, current_time: float) -> Dict[str, int]:
        """
        Get summary of fleet state.

        Args:
            current_time: Current simulation time

        Returns:
            Dictionary with counts by state
        """
        return {
            "total": len(self.nodes),
            "unavailable": len(self.get_nodes_by_state(NodeState.UNAVAILABLE)),
            "idle": len(self.get_nodes_by_state(NodeState.IDLE)),
            "running": len(self.get_nodes_by_state(NodeState.RUNNING)),
            "utilization": self.get_fleet_utilization(current_time),
        }
