"""
Earliest Deadline First (EDF) Scheduler for CGRSP

Assigns jobs with the earliest deadlines first to available nodes.
Prioritizes faster GPUs for jobs with tighter deadlines.
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus

class EDFScheduler(Scheduler):
    """
    Earliest Deadline First (EDF) scheduler.

    This scheduler prioritizes jobs that are closest to their deadlines.
    It attempts to minimize deadline misses by processing urgent jobs first.
    
    Cloud Awareness:
    - Checks StockStatus (High/Low) to avoid high-latency provisioning.
    - If stock is Low for a GPU type, it considers alternatives.
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

        # Sort jobs by deadline (earliest first)
        sorted_jobs = sorted(jobs, key=lambda j: j.deadline)
        
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

        used_nodes = set()
        for job in sorted_jobs:
            capable = [n for n in available_nodes if n.node_id not in used_nodes and n.can_accept_job(job.required_memory)]

            if not capable:
                assignments[job.job_id] = None
                continue

            # Select fastest node for this job's complexity, stock-availability tiered.
            # Use actual execution time (not a generic speed multiplier) so that the
            # correct node is chosen per complexity level, e.g. RTX A4500 (88.3 s)
            # beats RTX 3090 (91.2 s) for high-complexity jobs.
            def node_preference(n, _job=job):
                status = stock_info.get(n.resource_type, StockStatus.HIGH)
                exec_time = self.get_execution_time(_job, n)
                if status == StockStatus.HIGH:
                    return (0, exec_time)
                elif status == StockStatus.MEDIUM:
                    return (1, exec_time)
                return (2, exec_time)

            capable.sort(key=node_preference)

            selected_node = capable[0]
            assignments[job.job_id] = selected_node.node_id
            used_nodes.add(selected_node.node_id)

        return assignments

def create_edf_scheduler(fleet_manager, seed: Optional[int] = None) -> EDFScheduler:
    return EDFScheduler(fleet_manager, seed)
