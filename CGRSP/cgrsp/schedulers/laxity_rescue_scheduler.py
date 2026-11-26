"""
SPT with Laxity Rescue Scheduler for CGRSP

Uses Shortest Processing Time (SPT) by default for throughput, but promotes
any job whose laxity drops below a threshold to an urgent tier dispatched
with Earliest Deadline First (EDF) ordering.

Laxity of job j at time t:
    L_j(t) = deadline_j - t - estimated_render_time_j

When L_j(t) < rescue_threshold a miss is imminent unless the job is
dispatched immediately.  The scheduler splits the queue into:
  Tier 1 (urgent):    L_j < rescue_threshold  → sorted by earliest deadline
  Tier 2 (normal):    L_j >= rescue_threshold → sorted by shortest processing time

Jobs in Tier 1 are always dispatched before Tier 2.
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus


class LaxityRescueScheduler(Scheduler):
    """
    SPT + laxity-based deadline rescue.

    Parameters
    ----------
    rescue_threshold : float
        Laxity (seconds) below which a job is promoted to urgent tier.
        Default 600 s (10 min).  Use 300 s for a tighter (5-min) trigger.
    cost_weight : float
        Cost vs speed tradeoff for node selection within a tier (0–1).
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None,
                 rescue_threshold: float = 600.0,
                 cost_weight: float = 0.3):
        super().__init__(fleet_manager, seed)
        self.rescue_threshold = rescue_threshold
        self.cost_weight = max(0.0, min(1.0, cost_weight))

    # ── public API ────────────────────────────────────────────────────────────

    def schedule(
        self,
        jobs: List[Job],
        current_time: float,
    ) -> Dict[int, Optional[int]]:
        assignments: Dict[int, Optional[int]] = {}
        if not jobs:
            return assignments

        available_nodes = self.get_idle_nodes()
        if not available_nodes:
            for job in jobs:
                assignments[job.job_id] = None
            return assignments

        stock_info = self._get_stock_info()

        # Pre-compute min laxity for each job (using fastest available node).
        # If no capable node exists, min_exec = inf and laxity = -inf
        # (job cannot be dispatched at all this round).
        job_meta = []
        for job in jobs:
            min_exec = self._min_exec_time(job, available_nodes)
            if min_exec == float('inf'):
                laxity = float('-inf')  # No capable node — skip assignment
            else:
                laxity = job.deadline - current_time - min_exec
            job_meta.append((job, min_exec, laxity))

        # Split into urgent (rescue) and normal tiers
        urgent = [(j, e, l) for j, e, l in job_meta if l < self.rescue_threshold]
        normal = [(j, e, l) for j, e, l in job_meta if l >= self.rescue_threshold]

        # Sort: urgent by earliest absolute deadline; normal by shortest exec time
        urgent.sort(key=lambda x: x[0].deadline)
        normal.sort(key=lambda x: x[1])

        ordered = urgent + normal
        remaining_nodes = list(available_nodes)

        for job, _, _ in ordered:
            if not remaining_nodes:
                assignments[job.job_id] = None
                continue
            capable = [n for n in remaining_nodes
                       if n.can_accept_job(job.required_memory)]
            if not capable:
                assignments[job.job_id] = None
                continue
            node = self._select_node(job, capable, stock_info)
            if node:
                assignments[job.job_id] = node.node_id
                remaining_nodes.remove(node)
            else:
                assignments[job.job_id] = None

        return assignments

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_stock_info(self) -> dict:
        stock_info = {}
        if hasattr(self.fleet_manager, 'get_stock_status'):
            from cgrsp.config import ResourceType
            for rt in ResourceType.supported_gpus():
                stock_info[rt] = self.fleet_manager.get_stock_status(rt)
        return stock_info

    def _min_exec_time(self, job: Job, nodes: List[Node]) -> float:
        best = float('inf')
        for node in nodes:
            try:
                if node.can_accept_job(job.required_memory):
                    t = self.get_execution_time(job, node)
                    if t < best:
                        best = t
            except Exception:
                pass
        return best  # float('inf') if no capable node — caller handles this

    def _select_node(self, job: Job, capable: List[Node], stock_info: dict
                     ) -> Optional[Node]:
        scores = []
        for node in capable:
            try:
                exec_time = self.get_execution_time(job, node)
                fastest   = min(self.get_execution_time(job, n) for n in capable)
                cheapest  = min(n.get_per_second_cost() for n in capable)
                norm_t    = exec_time / fastest if fastest > 0 else 1.0
                norm_c    = node.get_per_second_cost() / cheapest if cheapest > 0 else 1.0
                penalty   = {StockStatus.HIGH: 0.0,
                             StockStatus.MEDIUM: 0.2,
                             StockStatus.LOW: 1.0}.get(
                    stock_info.get(node.resource_type, StockStatus.HIGH), 0.0)
                score = ((1.0 - self.cost_weight) * norm_t
                         + self.cost_weight * norm_c
                         + penalty)
                scores.append((node, score))
            except Exception:
                pass
        if not scores:
            return None
        scores.sort(key=lambda x: x[1])
        return scores[0][0]

    def get_scheduler_info(self) -> dict:
        return {
            "name": "LaxityRescue",
            "description": (
                f"SPT with EDF rescue below {self.rescue_threshold:.0f}s laxity "
                f"({self.rescue_threshold/60:.0f}-min window)"
            ),
            "rescue_threshold_s": self.rescue_threshold,
            "cost_weight": self.cost_weight,
        }
