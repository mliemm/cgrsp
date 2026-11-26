"""
HEFT-derived LPT baseline for CGRSP.

The Heterogeneous Earliest Finish Time (HEFT) heuristic of Topcuoglu, Hariri
and Wu (2002) schedules a task graph in two phases. It first ranks tasks by
upward rank, the longest path from the task to the sink through mean node and
edge costs, and then places them, in rank order, on the processor with the
earliest finish time. Both phases are defined on a task graph.

The CGRSP workload is a set of independent jobs, so the graph reduces to
isolated nodes and the upward rank of every job is exactly its mean execution
time over the fleet's GPU types. The HEFT phases then become:

1. Order jobs by decreasing mean execution time, which is the classical
   Longest Processing Time (LPT) list-scheduling rule.
2. Give each job, in that order, the idle slot with the earliest finish time
   current_time + W(kappa_j, g); ties go to the lower slot id.

This is the standard HEFT reduction for independent jobs, and it is
deliberately stock-agnostic: the rank and the finish-time comparison use only
the empirical mean execution time W(kappa, g). Neither the fleet's qualitative
stock status nor the provisioning delay a dispatch will actually incur enters
the decision, which matches the classical heuristics' assumption of a fixed,
known machine and contrasts directly with the stock-aware policies (SPT, EDF,
RH, CADR).

Difference from SPT (spt_scheduler.py), which can look superficially similar:
SPT orders jobs shortest first by the minimum execution time over the idle
nodes and then chooses a node with a cost-aware blend of normalised time,
normalised price and a stock-status penalty. HEFT-LPT orders jobs longest
first by the fleet mean and chooses the earliest-finish node.
"""

from typing import Dict, List, Optional

from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node


class HEFTLPTScheduler(Scheduler):
    """
    HEFT reduced to independent jobs: LPT ordering plus earliest-finish placement.

    Parameters
    ----------
    fleet_manager : FleetManager
        Fleet used to enumerate the GPU types that define the mean execution
        time of each job.
    seed : int, optional
        Unused (the policy is deterministic); kept for interface consistency.
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None):
        super().__init__(fleet_manager, seed)
        # Mean execution time depends only on the integer complexity class, so
        # one value per class is cached for the whole run. Per (complexity,
        # resource type) execution times are memoised for the same reason: the
        # model is deterministic, and placement asks for them on every call.
        self._mean_cache: Dict[int, float] = {}
        self._exec_cache: Dict[tuple, float] = {}

    def _exec(self, job: Job, node: Node) -> float:
        """Memoised self.get_execution_time(job, node)."""
        key = (int(job.complexity), node.resource_type)
        cached = self._exec_cache.get(key)
        if cached is None:
            cached = self.get_execution_time(job, node)
            self._exec_cache[key] = cached
        return cached

    def _fleet_mean_execution_time(self, job: Job) -> float:
        """
        Mean execution time of a job over the GPU types present in the fleet.

        This is the upward rank of an independent job: one node, no successors,
        so the rank is exactly the average node cost. The mean is taken over
        the distinct resource types of the fleet (one representative node each),
        not over slots, so a 4-type fleet with a duplicated slot still weighs
        each type once.
        """
        complexity = int(job.complexity)
        cached = self._mean_cache.get(complexity)
        if cached is not None:
            return cached

        representative: Dict[object, Node] = {}
        for node in self.fleet_manager.get_all_nodes():
            representative.setdefault(node.resource_type, node)

        if not representative:
            mean = float(job.estimated_duration)
        else:
            times = [
                self._exec(job, node)
                for node in representative.values()
            ]
            mean = sum(times) / len(times)

        self._mean_cache[complexity] = mean
        return mean

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """
        Assign queued jobs by LPT order to the currently idle slots.

        Every queued job receives a key; jobs that find no idle capable slot in
        this round map to None and stay in the queue.
        """
        assignments: Dict[int, Optional[int]] = {job.job_id: None for job in jobs}
        if not jobs:
            return assignments

        available = sorted(self.get_idle_nodes(), key=lambda node: node.node_id)
        if not available:
            return assignments

        ordered = sorted(
            jobs,
            key=lambda job: (-self._fleet_mean_execution_time(job), job.job_id),
        )

        for job in ordered:
            if not available:
                break
            capable = [
                node for node in available
                if node.can_accept_job(job.required_memory)
            ]
            if not capable:
                continue
            node = min(
                capable,
                key=lambda n: (
                    current_time + self._exec(job, n),
                    n.node_id,
                ),
            )
            assignments[job.job_id] = node.node_id
            available.remove(node)

        return assignments


def create_heft_lpt_scheduler(fleet_manager, seed: Optional[int] = None) -> HEFTLPTScheduler:
    """Factory function to create a HEFTLPTScheduler."""
    return HEFTLPTScheduler(fleet_manager, seed)
