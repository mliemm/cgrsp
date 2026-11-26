"""
Shortest Processing Time (SPT) Scheduler for CGRSP

Implements SPT scheduling policy with cost-aware tiebreaking.
Jobs are prioritized by shortest processing time first, with cost
optimization for jobs with similar durations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node, FleetManager
from cgrsp.data.execution_times import ExecutionTimeModel, GPUType
from cgrsp.data.runpod_availability import StockStatus


@dataclass
class JobSPTInfo:
    """Information about a job for SPT scheduling."""
    job: Job
    estimated_time: float
    cost_per_second: float
    total_cost: float
    urgency: float  # deadline proximity factor


class SPTScheduler(Scheduler):
    """
    Shortest Processing Time (SPT) scheduler with cost optimization.

    This scheduler prioritizes jobs with shorter processing times to improve
    throughput and reduce average waiting time. When jobs have similar
    processing times, it uses cost-aware tiebreaking.

    Algorithm:
    1. For each job, calculate estimated execution time on available GPUs
    2. Sort jobs by shortest processing time (primary criteria)
    3. Assign jobs to available GPUs by a speed-weighted score that blends
       normalised execution time with normalised price (cost_weight = 0.3)

    Advantages:
    - Reduces average waiting time (short jobs finish quickly)
    - Improves throughput (more jobs completed per time unit)
    - Cost-aware optimization for similar-duration jobs
    - Good for interactive/low-latency workloads

    Disadvantages:
    - May starve long-running jobs (convoy effect)
    - Doesn't consider deadlines directly
    - Requires accurate execution time estimation
    - May increase deadline misses for long jobs

    Cost Optimization:
    - Node choice uses the blended score unconditionally, so a cheaper GPU
      wins whenever its normalised slowdown is smaller than its price saving
    - Balances speed vs. cost for efficient resource utilization
    """

    def __init__(
        self,
        fleet_manager: FleetManager,
        seed: Optional[int] = None,
        cost_weight: float = 0.3
    ):
        """
        Initialize SPT scheduler.

        Args:
            fleet_manager: Fleet manager for accessing GPU nodes
            seed: Random seed for reproducibility
            cost_weight: Weight for cost in the node-selection blend (0.0-1.0)
                        Higher = more cost-conscious
        """
        super().__init__(fleet_manager, seed)
        self.cost_weight = max(0.0, min(1.0, cost_weight))

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        assignments = {}

        if not jobs:
            return assignments

        # Get available idle nodes
        idle_nodes = self.get_idle_nodes()

        if not idle_nodes:
            # No nodes available, all jobs wait
            for job in jobs:
                assignments[job.job_id] = None
            return assignments

        # Get qualitative stock status from fleet manager (no counts)
        stock_info = {}
        if hasattr(self.fleet_manager, 'get_stock_status'):
            from cgrsp.config import ResourceType
            for rt in ResourceType.supported_gpus():
                stock_info[rt] = self.fleet_manager.get_stock_status(rt)

        # Calculate SPT priority for each job
        job_priorities = []
        for job in jobs:
            # Calculate estimated execution time on each available GPU
            # and use the minimum as the job's "processing time"
            min_exec_time = float('inf')
            selected_node = None

            for node in idle_nodes:
                try:
                    if node.can_accept_job(job.required_memory):
                        exec_time = self.get_execution_time(job, node)
                        if exec_time < min_exec_time:
                            min_exec_time = exec_time
                            selected_node = node
                except Exception as e:
                    continue

            # If no capable node found, use a large time
            if min_exec_time == float('inf'):
                min_exec_time = 999999.0

            job_priorities.append({
                'job': job,
                'exec_time': min_exec_time,
                'preferred_node': selected_node
            })

        # Sort by execution time (SPT primary criteria) - shortest first
        job_priorities.sort(key=lambda x: x['exec_time'])

        # Process jobs in SPT order
        available_nodes = list(idle_nodes)  # Create mutable copy

        for priority_info in job_priorities:
            job = priority_info['job']

            if not available_nodes:
                # No more nodes available
                assignments[job.job_id] = None
                continue

            # Find capable nodes
            required_memory = job.required_memory
            capable_nodes = [
                node for node in available_nodes
                if node.can_accept_job(required_memory)
            ]

            if not capable_nodes:
                # No capable nodes available
                assignments[job.job_id] = None
                continue

            # Select best node using cost-aware optimization
            selected_node = self._select_node_with_cost_awareness(
                job,
                capable_nodes,
                stock_info
            )

            if selected_node:
                assignments[job.job_id] = selected_node.node_id
                available_nodes.remove(selected_node)
            else:
                assignments[job.job_id] = None

        return assignments

    def _select_node_with_cost_awareness(
        self,
        job: Job,
        capable_nodes: List[Node],
        stock_info: Dict
    ) -> Optional[Node]:
        if not capable_nodes:
            return None

        # Calculate execution time and cost for each capable node
        node_scores = []
        for node in capable_nodes:
            try:
                exec_time = self.get_execution_time(job, node)
                cost_per_sec = node.get_per_second_cost()
                total_cost = exec_time * cost_per_sec

                # Score: lower is better
                # Primary factor: execution time (SPT)
                # Secondary factor: cost (weighted by cost_weight)

                # Normalize time relative to fastest option
                fastest_time = min(
                    self.get_execution_time(job, n)
                    for n in capable_nodes
                )
                normalized_time = exec_time / fastest_time

                # Normalize cost relative to cheapest option
                cheapest_cost = min(n.get_per_second_cost() for n in capable_nodes)
                normalized_cost = cost_per_sec / cheapest_cost

                # Combine score (lower is better)
                # Add a penalty for non-HIGH stock (normalized to typical time/cost scales)
                status = stock_info.get(node.resource_type, StockStatus.HIGH)
                status_penalty = {
                    StockStatus.HIGH: 0.0,
                    StockStatus.MEDIUM: 0.2,
                    StockStatus.LOW: 1.0,
                }.get(status, 0.0)

                score = (
                    (1.0 - self.cost_weight) * normalized_time +
                    self.cost_weight * normalized_cost +
                    status_penalty
                )

                node_scores.append((node, score, exec_time, total_cost))

            except Exception:
                continue

        if not node_scores:
            return None


        # Sort by score (lower is better)
        node_scores.sort(key=lambda x: x[1])

        # Return best node
        return node_scores[0][0]

    def get_scheduler_info(self) -> Dict:
        """Get information about the scheduler configuration."""
        return {
            "name": "SPT",
            "description": "Shortest Processing Time with cost optimization",
            "cost_weight": self.cost_weight,
            "prioritization": "Shortest jobs first, cost-aware tiebreaking"
        }


def create_spt_scheduler(
    fleet_manager: FleetManager,
    seed: Optional[int] = None,
    cost_weight: float = 0.3
) -> SPTScheduler:
    """
    Factory function to create an SPT scheduler.

    Args:
        fleet_manager: Fleet manager instance
        seed: Optional random seed
        cost_weight: Cost optimization weight (0.0-1.0, default 0.3)

    Returns:
        Configured SPTScheduler instance
    """
    return SPTScheduler(
        fleet_manager=fleet_manager,
        seed=seed,
        cost_weight=cost_weight
    )