"""
Random Scheduler for CGRSP

Assigns jobs to random available GPU nodes. Useful as a baseline for comparison.
"""

import random
from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node

class RandomScheduler(Scheduler):
    """
    Random scheduler.

    Assigns jobs to available GPU nodes in random order.
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None):
        super().__init__(fleet_manager, seed)
        if seed is not None:
            random.seed(seed)

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        assignments = {}
        
        if not jobs:
            return assignments

        # Shuffle jobs for random processing order
        random_jobs = list(jobs)
        random.shuffle(random_jobs)
        
        available_nodes = self.get_available_nodes()
        if not available_nodes:
            for job in random_jobs:
                assignments[job.job_id] = None
            return assignments

        used_nodes = set()
        for job in random_jobs:
            capable = [n for n in available_nodes if n.node_id not in used_nodes and n.can_accept_job(job.required_memory)]
            
            if not capable:
                assignments[job.job_id] = None
                continue

            # Select a random capable node
            selected_node = random.choice(capable)
            assignments[job.job_id] = selected_node.node_id
            used_nodes.add(selected_node.node_id)

        return assignments

def create_random_scheduler(fleet_manager, seed: Optional[int] = None) -> RandomScheduler:
    return RandomScheduler(fleet_manager, seed)
