"""
Balanced Scheduler for CGRSP

Balances between performance (Shortest Processing Time) and cost (Lowest Cost First).
Uses a weight parameter alpha to adjust the trade-off.
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus

class BalancedScheduler(Scheduler):
    """
    Balanced scheduler.

    This scheduler balances execution speed and operational cost using
    a weighted scoring mechanism.
    
    Cloud Awareness:
    - Incorporates StockStatus (High/Low) as a "provisioning penalty".
    - Score = alpha * Norm_Time + (1-alpha) * Norm_Cost + Status_Penalty
    """

    def __init__(self, fleet_manager, alpha: float = 0.8, seed: Optional[int] = None):
        """
        Initialize the balanced scheduler.

        Args:
            fleet_manager: Fleet manager for accessing GPU nodes
            alpha: Weight for performance (0.0 to 1.0).
                  1.0 = Performance only (SPT), 0.0 = Cost only (LCF).
                  Default 0.8: speed-biased balance — prefers RTX_3090 over RTX_A4500
                  (differs from LCF which ranks by total job cost: A5000<A4500<3090<A4000).
            seed: Random seed
        """
        super().__init__(fleet_manager, seed)
        self.alpha = alpha
        
        # Penalties for stock status (normalized scale)
        self.status_penalties = {
            StockStatus.HIGH: 0.0,
            StockStatus.MEDIUM: 0.2, # Slight penalty
            StockStatus.LOW: 1.0,    # Significant penalty for potential queue
        }

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        assignments = {}
        
        if not jobs:
            return assignments

        # Keep FIFO submission order so Balanced is distinct from EDF
        sorted_jobs = sorted(jobs, key=lambda j: j.submit_time)
        
        available_nodes = self.get_available_nodes()
        if not available_nodes:
            for job in sorted_jobs:
                assignments[job.job_id] = None
            return assignments

        # Get qualitative stock status from fleet manager (no counts)
        stock_info = {}
        if hasattr(self.fleet_manager, 'get_stock_status'):
            from cgrsp.config import ResourceType
            for rt in ResourceType.supported_gpus():
                stock_info[rt] = self.fleet_manager.get_stock_status(rt)

        # Global normalization bounds — computed once per schedule() call so that
        # the same scale is used for every (job, node) pair.  Range-based
        # normalisation with only 2 available nodes produces identical scores for
        # the fastest/cheapest and slowest/cheapest nodes (both → 0.5), causing
        # Balanced to degenerate into the same decisions as EDF.
        all_fleet_nodes = self.fleet_manager.get_all_nodes()
        global_max_cost = max(
            (n.get_per_second_cost() for n in all_fleet_nodes), default=1.0
        )

        used_nodes = set()
        for job in sorted_jobs:
            capable = [n for n in available_nodes if n.node_id not in used_nodes and n.can_accept_job(job.required_memory)]

            if not capable:
                assignments[job.job_id] = None
                continue

            # Per-job global max exec time (depends on complexity)
            global_max_time = max(
                (self.get_execution_time(job, n) for n in all_fleet_nodes), default=1.0
            )

            def get_score(node, _gmt=global_max_time, _gmc=global_max_cost, _job=job):
                t = self.get_execution_time(_job, node)
                c = node.get_per_second_cost()
                status = stock_info.get(node.resource_type, StockStatus.HIGH)
                penalty = self.status_penalties.get(status, 0.0)

                # Normalise by global fleet max — prevents symmetric ties that
                # occur when range-normalisation is applied to only 2 nodes.
                norm_t = t / _gmt
                norm_c = c / _gmc

                return self.alpha * norm_t + (1.0 - self.alpha) * norm_c + penalty

            # Sort by score (lower is better)
            capable.sort(key=get_score)
            
            selected_node = capable[0]
            assignments[job.job_id] = selected_node.node_id
            used_nodes.add(selected_node.node_id)

        return assignments

def create_balanced_scheduler(fleet_manager, alpha: float = 0.8, seed: Optional[int] = None) -> BalancedScheduler:
    return BalancedScheduler(fleet_manager, alpha, seed)
