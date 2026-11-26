"""
Base Scheduler Interface for CGRSP

Abstract base class for all scheduling algorithms in the CGRSP simulator.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from cgrsp.simulator.queue import Job, JobQueue
from cgrsp.simulator.fleet import Node, FleetManager
from cgrsp.data.execution_times import ExecutionTimeModel, GPUType


class Scheduler(ABC):
    """
    Abstract base class for all CGRSP scheduling algorithms.

    All schedulers must implement the `schedule` method which takes
    a list of jobs and a fleet manager, and returns assignment decisions.
    """

    def __init__(self, fleet_manager: FleetManager, seed: Optional[int] = None):
        """
        Initialize the scheduler.

        Args:
            fleet_manager: Fleet manager for accessing GPU nodes
            seed: Random seed for reproducibility (where applicable)
        """
        self.fleet_manager = fleet_manager
        self.seed = seed

    @abstractmethod
    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """
        Make scheduling decisions for a list of jobs.

        Args:
            jobs: List of jobs to schedule (from job queue)
            current_time: Current simulation time

        Returns:
            Dictionary mapping job_id to node_id (or None if job waits)
        """
        pass

    def get_idle_nodes(self) -> List[Node]:
        """Get all idle nodes in the fleet."""
        return self.fleet_manager.get_idle_nodes()

    def get_available_nodes(
        self,
        required_memory: Optional[float] = None
    ) -> List[Node]:
        """
        Get nodes that can accept new jobs.

        Args:
            required_memory: Optional VRAM requirement in GB

        Returns:
            List of available nodes
        """
        return self.fleet_manager.get_available_nodes(required_memory)

    def get_execution_time(self, job: Job, node: Node) -> float:
        """
        Get execution time for a job on a specific node.

        Args:
            job: Job to schedule
            node: Target node

        Returns:
            Execution time in seconds
        """
        # Map ResourceType to GPUType
        gpu_type_map = {
            "RTX_3090": GPUType.RTX_3090,
            "RTX_A5000": GPUType.RTX_A5000,
            "RTX_A4500": GPUType.RTX_A4500,
            "RTX_A4000": GPUType.RTX_A4000,
        }

        gpu_type = gpu_type_map.get(node.resource_type.value, GPUType.RTX_3090)
        complexity = int(job.complexity)

        return ExecutionTimeModel.get_execution_time(complexity, gpu_type)

    def get_job_cost(self, job: Job, node: Node) -> float:
        """
        Calculate operational cost for running a job on a node.

        Args:
            job: Job to schedule
            node: Target node

        Returns:
            Total cost in USD
        """
        execution_time = self.get_execution_time(job, node)
        per_second_cost = node.get_per_second_cost()
        return execution_time * per_second_cost

    def can_meet_deadline(
        self,
        job: Job,
        node: Node,
        current_time: float
    ) -> bool:
        """
        Check if job can meet its deadline on a given node.

        Args:
            job: Job to check
            node: Target node
            current_time: Current simulation time

        Returns:
            True if job can complete before deadline
        """
        execution_time = self.get_execution_time(job, node)
        completion_time = current_time + execution_time
        return completion_time <= job.deadline

    def get_nodes_by_resource_type(
        self,
        resource_type: str,
        state: Optional[str] = None
    ) -> List[Node]:
        """
        Get nodes of a specific resource type.

        Args:
            resource_type: GPU resource type (e.g., "RTX_3090")
            state: Optional state filter ("idle", "running", "unavailable")

        Returns:
            List of matching nodes
        """
        from cgrsp.config import ResourceType
        from cgrsp.simulator.fleet import NodeState

        rt = ResourceType(resource_type)
        ns = NodeState(state) if state else None

        return self.fleet_manager.get_nodes_by_resource_type(rt, ns)


class SchedulerMetrics:
    """
    Tracks scheduler performance metrics.

    Attributes:
        total_jobs_scheduled: Total number of jobs scheduled
        total_jobs_waited: Number of jobs that had to wait
        assignments_by_gpu_type: Count of assignments per GPU type
        total_cost: Total operational cost of all assignments
        deadline_misses: Number of jobs that missed deadlines
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_jobs_scheduled = 0
        self.total_jobs_waited = 0
        self.assignments_by_gpu_type: Dict[str, int] = {}
        self.total_cost = 0.0
        self.deadline_misses = 0
        self.scheduling_calls = 0

    def record_assignment(
        self,
        job: Job,
        node: Optional[Node],
        current_time: float
    ):
        """
        Record a job assignment decision.

        Args:
            job: Job that was assigned (or not)
            node: Node assigned to (None if job waits)
            current_time: Current simulation time
        """
        self.scheduling_calls += 1

        if node is None:
            self.total_jobs_waited += 1
            return

        self.total_jobs_scheduled += 1

        gpu_type = node.resource_type.value
        self.assignments_by_gpu_type[gpu_type] = (
            self.assignments_by_gpu_type.get(gpu_type, 0) + 1
        )

        cost = self._calculate_job_cost(job, node)
        self.total_cost += cost

        # Check deadline
        if not self._can_meet_deadline(job, node, current_time):
            self.deadline_misses += 1

    def _calculate_job_cost(self, job: Job, node: Node) -> float:
        """Calculate job cost (helper method)."""
        execution_time = self._get_execution_time(job, node)
        per_second_cost = node.get_per_second_cost()
        return execution_time * per_second_cost

    def _get_execution_time(self, job: Job, node: Node) -> float:
        """Get execution time (helper method)."""
        from cgrsp.data.execution_times import ExecutionTimeModel, GPUType

        gpu_type_map = {
            "RTX_3090": GPUType.RTX_3090,
            "RTX_A5000": GPUType.RTX_A5000,
            "RTX_A4500": GPUType.RTX_A4500,
            "RTX_A4000": GPUType.RTX_A4000,
        }

        gpu_type = gpu_type_map.get(node.resource_type.value, GPUType.RTX_3090)
        complexity = int(job.complexity)

        return ExecutionTimeModel.get_execution_time(complexity, gpu_type)

    def _can_meet_deadline(
        self,
        job: Job,
        node: Node,
        current_time: float
    ) -> bool:
        """Check if job can meet deadline (helper method)."""
        execution_time = self._get_execution_time(job, node)
        completion_time = current_time + execution_time
        return completion_time <= job.deadline

    def get_metrics(self) -> Dict:
        """
        Get current metrics summary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "total_jobs_scheduled": self.total_jobs_scheduled,
            "total_jobs_waited": self.total_jobs_waited,
            "scheduling_calls": self.scheduling_calls,
            "assignments_by_gpu_type": self.assignments_by_gpu_type.copy(),
            "total_cost": self.total_cost,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": (
                self.deadline_misses / self.total_jobs_scheduled
                if self.total_jobs_scheduled > 0 else 0.0
            ),
            "avg_cost_per_job": (
                self.total_cost / self.total_jobs_scheduled
                if self.total_jobs_scheduled > 0 else 0.0
            ),
            "avg_assignments_per_call": (
                self.total_jobs_scheduled / self.scheduling_calls
                if self.scheduling_calls > 0 else 0.0
            ),
        }
