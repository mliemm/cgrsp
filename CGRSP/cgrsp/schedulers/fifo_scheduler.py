"""
First-In-First-Out (FIFO) Scheduler for CGRSP

Implements the simplest scheduling policy where jobs are processed
in the exact order they arrive, regardless of complexity, deadlines,
or other factors.
"""

from typing import Dict, List, Optional

from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node


class FIFOScheduler(Scheduler):
    """
    First-In-First-Out (FIFO) scheduler.

    This is the simplest possible scheduling policy that processes jobs
    in strict submission order. Jobs are assigned to available GPU nodes
    in the order they arrived at the system.

    Algorithm:
    1. Sort jobs by submission time (earliest first)
    2. For each job in order:
       - If no idle nodes available, job waits
       - Select first available idle node
       - Assign job to node, mark node as busy
    3. Return assignment decisions

    Advantages:
    - Simple and predictable behavior
    - No starvation (every job eventually gets served)
    - Easy to implement and debug
    - Fair in terms of waiting order
    - Good baseline for comparison

    Disadvantages:
    - Ignores job complexity and deadlines
    - May lead to poor deadline performance
    - Doesn't optimize for cost or speed
    - No priority consideration

    This serves as an important baseline to compare against more
    sophisticated scheduling algorithms.
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None):
        """
        Initialize FIFO scheduler.

        Args:
            fleet_manager: Fleet manager for accessing GPU nodes
            seed: Random seed (not used in FIFO but kept for interface consistency)
        """
        super().__init__(fleet_manager, seed)

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """
        Schedule jobs using FIFO policy (first-in-first-out).

        Jobs are processed in strict submission order, regardless of
        complexity, priority, or deadlines.

        Args:
            jobs: List of jobs from the job queue
            current_time: Current simulation time

        Returns:
            Dictionary {job_id: node_id} or None if job waits
        """
        assignments = {}

        # Sort jobs by submission time (FIFO order)
        # Use submit_time as the primary sort key
        sorted_jobs = sorted(jobs, key=lambda job: job.submit_time)

        # Get available idle nodes
        idle_nodes = self.get_idle_nodes()

        if not idle_nodes:
            # No nodes available, all jobs wait
            for job in sorted_jobs:
                assignments[job.job_id] = None
            return assignments

        # Process jobs in FIFO order
        available_nodes = list(idle_nodes)  # Create mutable copy

        for job in sorted_jobs:
            # Check if job has memory requirements
            required_memory = job.required_memory

            # Filter nodes that can accept this job
            capable_nodes = [
                node for node in available_nodes
                if node.can_accept_job(required_memory)
            ]

            if not capable_nodes:
                # No capable nodes available, job waits
                assignments[job.job_id] = None
                continue

            # Select first available node (FIFO for node selection too)
            selected_node = capable_nodes[0]

            # Assign job to node
            assignments[job.job_id] = selected_node.node_id

            # Remove node from available pool
            available_nodes.remove(selected_node)

        return assignments


def create_fifo_scheduler(fleet_manager, seed: Optional[int] = None) -> FIFOScheduler:
    """
    Factory function to create a FIFOScheduler.

    Args:
        fleet_manager: Fleet manager instance
        seed: Optional random seed (not used but kept for interface consistency)

    Returns:
        Configured FIFOScheduler instance
    """
    return FIFOScheduler(fleet_manager, seed)