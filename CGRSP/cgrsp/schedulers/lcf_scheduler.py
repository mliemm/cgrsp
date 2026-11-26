"""
Lowest Cost First (LCF) Scheduler for CGRSP

Assigns jobs to the cheapest available GPUs first to minimize operational cost.
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus

class LCFScheduler(Scheduler):
    """
    Lowest Cost First (LCF) scheduler.

    This scheduler prioritizes cost reduction by assigning jobs to the 
    cheapest available resources first.
    
    Cloud Awareness:
    - Checks StockStatus (High/Low).
    - Avoids nodes with Low/Empty stock to minimize hidden provisioning costs.
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None):
        super().__init__(fleet_manager, seed)

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        assignments = {}
        
        if not jobs:
            return assignments

        # Sort jobs by submission time
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

        # Proportional stock penalties — total job cost stays the primary criterion
        STOCK_PENALTY = {
            StockStatus.HIGH: 0.00,
            StockStatus.MEDIUM: 0.05,
            StockStatus.LOW: 0.15,
        }

        used_nodes = set()
        for job in sorted_jobs:
            capable = [n for n in available_nodes if n.node_id not in used_nodes and n.can_accept_job(job.required_memory)]

            if not capable:
                assignments[job.job_id] = None
                continue

            def node_preference(n, _job=job):
                status = stock_info.get(n.resource_type, StockStatus.HIGH)
                # Use total job cost (exec_time × rate) — not just the hourly rate.
                # A4000 and A4500 share one hourly rate, so the rate alone cannot
                # separate them; total cost picks the A4000 at complexity 1 and 2
                # and the A4500 at complexity 3 (shorter mean render time).
                exec_time = self.get_execution_time(_job, n)
                total_cost = exec_time * n.get_per_second_cost()
                penalty = STOCK_PENALTY.get(status, 0.0)
                return total_cost * (1.0 + penalty)

            capable.sort(key=node_preference)

            selected_node = capable[0]
            assignments[job.job_id] = selected_node.node_id
            used_nodes.add(selected_node.node_id)

        return assignments

def create_lcf_scheduler(fleet_manager, seed: Optional[int] = None) -> LCFScheduler:
    return LCFScheduler(fleet_manager, seed)
