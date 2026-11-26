"""
Min-Min baseline for CGRSP.

Min-Min is the classical greedy list heuristic for independent jobs on
heterogeneous machines (Braun et al., 2001). Each round:

1. For every unscheduled job, find the idle node on which it would finish
   earliest, current_time + W(kappa_j, g).
2. Assign the job whose earliest completion time is the smallest, remove that
   node and that job, and repeat until no idle node or no schedulable job is
   left.

The pick is a single (completion time, job id, node id) tuple, so ties resolve
to the lower job id and then to the lower node id. Jobs that no idle node can
accept (for instance on VRAM grounds) are skipped and stay queued.

Min-Min favours short jobs that fit an idle slot quickly. Max-Min
(max_min_scheduler.py) is the same procedure with the pick inverted to the
largest earliest completion time, and the two are the standard pair used to
bracket the greedy family.
"""

from typing import Dict, List, Optional

from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job


class MinMinScheduler(Scheduler):
    """
    Min-Min: repeatedly assign the job with the smallest earliest completion time.

    Subclasses flip ``pick_largest`` for the Max-Min variant; the tie-breaking
    and node filtering live here only.
    """

    # False: pick the smallest earliest completion time (Min-Min).
    # True: pick the largest earliest completion time (Max-Min).
    pick_largest: bool = False

    def __init__(self, fleet_manager, seed: Optional[int] = None):
        super().__init__(fleet_manager, seed)
        # Execution times are deterministic per (complexity, resource type), and
        # every round asks for them again, so they are memoised for the run.
        self._exec_cache: Dict[tuple, float] = {}

    def _exec(self, job: Job, node) -> float:
        """Memoised self.get_execution_time(job, node)."""
        key = (int(job.complexity), node.resource_type)
        cached = self._exec_cache.get(key)
        if cached is None:
            cached = self.get_execution_time(job, node)
            self._exec_cache[key] = cached
        return cached

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """
        Assign queued jobs by repeated earliest-completion selection.

        Every queued job receives a key; unassigned jobs map to None.
        """
        assignments: Dict[int, Optional[int]] = {job.job_id: None for job in jobs}
        if not jobs:
            return assignments

        available = sorted(self.get_idle_nodes(), key=lambda node: node.node_id)
        remaining = list(jobs)

        while available and remaining:
            best_key = None
            best_job = None
            best_node = None

            for job in remaining:
                # Earliest-completion node for this job (ties: lower node id).
                node = None
                completion = None
                for candidate in available:
                    if not candidate.can_accept_job(job.required_memory):
                        continue
                    finish = current_time + self._exec(job, candidate)
                    if completion is None or (finish, candidate.node_id) < (completion, node.node_id):
                        completion = finish
                        node = candidate
                if node is None:
                    continue

                key = (
                    -completion if self.pick_largest else completion,
                    job.job_id,
                    node.node_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_job = job
                    best_node = node

            if best_job is None:
                break

            assignments[best_job.job_id] = best_node.node_id
            available.remove(best_node)
            remaining.remove(best_job)

        return assignments


def create_min_min_scheduler(fleet_manager, seed: Optional[int] = None) -> MinMinScheduler:
    """Factory function to create a MinMinScheduler."""
    return MinMinScheduler(fleet_manager, seed)
