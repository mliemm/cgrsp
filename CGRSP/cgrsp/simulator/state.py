"""
State Representation Module

Defines the observable state (scheduler view) and hidden state dynamics
(simulator internal) for the CGRSP simulator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from cgrsp.config import ResourceType


@dataclass
class NodeStatus:
    """Observable status of a single node."""
    node_id: int
    resource_type: ResourceType
    state: str  # "unavailable", "idle", "running"
    current_job_id: Optional[int] = None
    available_since: float = 0.0  # Time since state entry
    unavailable_reason: Optional[str] = None  # "customer", "maintenance", "failure"


@dataclass
class ObservableState:
    """
    Observable state of the system from the scheduler's perspective.

    This represents the information available to scheduling decisions at time t.
    """

    current_time: float = 0.0

    # Node fleet status
    node_statuses: Dict[int, NodeStatus] = field(default_factory=dict)

    # Cloud stock status (High/Medium/Low) - Qualitative knowledge only
    stock_statuses: Dict[ResourceType, str] = field(default_factory=dict)

    # Job queue (pending jobs)
    queued_jobs: List[int] = field(default_factory=list)

    # Temporal context
    hour_of_day: int = 0  # 0-23 from simulation time zero (no 08:00 offset; the availability model applies its own 08:00 anchor); no scheduler reads it
    day_of_week: int = 0  # 0-6 (Monday=0)

    # System statistics
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_deadline_missed: int = 0

    def get_available_nodes(self) -> List[NodeStatus]:
        """Get list of idle nodes available for scheduling."""
        return [
            status for status in self.node_statuses.values()
            if status.state == "idle"
        ]

    def get_unavailable_nodes(self) -> List[NodeStatus]:
        """Get list of unavailable nodes."""
        return [
            status for status in self.node_statuses.values()
            if status.state == "unavailable"
        ]

    def get_running_nodes(self) -> List[NodeStatus]:
        """Get list of nodes currently running jobs."""
        return [
            status for status in self.node_statuses.values()
            if status.state == "running"
        ]

    def get_node_count_by_state(self, state: str) -> int:
        """Count nodes in a specific state."""
        return sum(
            1 for status in self.node_statuses.values()
            if status.state == state
        )

    def get_nodes_by_resource_type(
        self,
        resource_type: ResourceType,
        state: Optional[str] = None
    ) -> List[NodeStatus]:
        """
        Get nodes of a specific resource type, optionally filtered by state.

        Args:
            resource_type: GPU resource type
            state: Optional state filter ("unavailable", "idle", "running")

        Returns:
            List of matching node statuses
        """
        nodes = [
            status for status in self.node_statuses.values()
            if status.resource_type == resource_type
        ]

        if state is not None:
            nodes = [s for s in nodes if s.state == state]

        return nodes

    def get_queue_length(self) -> int:
        """Get current number of jobs in queue."""
        return len(self.queued_jobs)

    def get_system_utilization(self) -> float:
        """
        Calculate current system utilization.

        Returns:
            Fraction of nodes in running state
        """
        total_nodes = len(self.node_statuses)
        if total_nodes == 0:
            return 0.0

        running = self.get_node_count_by_state("running")
        return running / total_nodes

    def clone(self) -> "ObservableState":
        """Create a deep copy of the observable state."""
        import copy

        cloned = ObservableState(
            current_time=self.current_time,
            hour_of_day=self.hour_of_day,
            day_of_week=self.day_of_week,
            total_jobs_submitted=self.total_jobs_submitted,
            total_jobs_completed=self.total_jobs_completed,
            total_jobs_deadline_missed=self.total_jobs_deadline_missed,
        )

        # Deep copy node statuses
        cloned.node_statuses = {
            node_id: NodeStatus(
                node_id=status.node_id,
                resource_type=status.resource_type,
                state=status.state,
                current_job_id=status.current_job_id,
                available_since=status.available_since,
                unavailable_reason=status.unavailable_reason,
            )
            for node_id, status in self.node_statuses.items()
        }

        # Copy job queue
        cloned.queued_jobs = self.queued_jobs.copy()

        return cloned

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "current_time": self.current_time,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "node_statuses": {
                node_id: {
                    "node_id": status.node_id,
                    "resource_type": status.resource_type.value,
                    "state": status.state,
                    "current_job_id": status.current_job_id,
                    "available_since": status.available_since,
                    "unavailable_reason": status.unavailable_reason,
                }
                for node_id, status in self.node_statuses.items()
            },
            "queued_jobs": self.queued_jobs,
            "total_jobs_submitted": self.total_jobs_submitted,
            "total_jobs_completed": self.total_jobs_completed,
            "total_jobs_deadline_missed": self.total_jobs_deadline_missed,
        }


@dataclass
class HiddenState:
    """
    Hidden state dynamics managed internally by the simulator.

    This represents information not directly observable to the scheduler,
    including Markov transition probabilities and future stochastic events.
    """

    # Pending stochastic transitions (scheduled but not yet executed)
    pending_transitions: Dict[int, str] = field(default_factory=dict)
    # Maps node_id -> transition_type (e.g., "to_customer", "to_maintenance")

    # Future event schedule (managed by simulator core)
    # This is a reference to the simulator's event queue

    # Markov chain state tracking
    last_transition_times: Dict[int, float] = field(default_factory=dict)
    # Maps node_id -> timestamp of last state transition

    # Transition counters (for analysis/debugging)
    transition_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Nested dict: from_state -> to_state -> count

    def record_transition(self, node_id: int, from_state: str, to_state: str) -> None:
        """
        Record a state transition for analysis.

        Args:
            node_id: Node identifier
            from_state: Previous state
            to_state: New state
        """
        if from_state not in self.transition_counts:
            self.transition_counts[from_state] = {}

        if to_state not in self.transition_counts[from_state]:
            self.transition_counts[from_state][to_state] = 0

        self.transition_counts[from_state][to_state] += 1

    def get_transition_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get the transition count matrix.

        Returns:
            Nested dict: from_state -> to_state -> count
        """
        return self.transition_counts.copy()

    def reset(self) -> None:
        """Reset all hidden state tracking."""
        self.pending_transitions.clear()
        self.last_transition_times.clear()
        self.transition_counts.clear()
