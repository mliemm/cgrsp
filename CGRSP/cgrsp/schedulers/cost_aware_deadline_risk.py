"""
Cost-Aware Deadline-Risk (CADR) Scheduler for CGRSP.

Combines two ideas:
  (A) Adaptive SPT/EDF ordering by a scale-free critical ratio
        CR_j = (deadline_j - t) / e_j
      where e_j is the fastest feasible mean execution time. Jobs are
      tiered as doomed (CR <= 1, demoted), at-risk (1 < CR <= CR*, EDF
      order), or safe (CR > CR*, SPT order).
  (B) Cost-aware node selection: among nodes that still meet the deadline
      with a margin, pick the cheapest; otherwise pick the fastest.

Unlike SPT+Rescue's absolute laxity threshold (theta seconds), CR* is
dimensionless and transfers across deadline tightness and load. Unlike
the speed-primary placement of every other scheduler, component (B) banks
the cheap-GPU saving on jobs that have slack. Set cost_aware=False to
isolate component (A) for the ablation (fastest-node placement).
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus

_DOOMED = 2
_AT_RISK = 0
_SAFE = 1
_NO_NODE = 3


class CostAwareDeadlineRiskScheduler(Scheduler):
    """Cost-aware deadline-risk scheduler (critical-ratio ordering + cost-first placement)."""

    def __init__(self, fleet_manager, seed: Optional[int] = None,
                 cr_star: float = 3.0,
                 margin: float = 0.0,
                 cost_aware: bool = True):
        super().__init__(fleet_manager, seed)
        self.cr_star = max(1.0, cr_star)
        self.margin = max(0.0, margin)
        self.cost_aware = cost_aware

    def schedule(self, jobs: List[Job], current_time: float) -> Dict[int, Optional[int]]:
        assignments: Dict[int, Optional[int]] = {}
        if not jobs:
            return assignments

        available_nodes = self.get_idle_nodes()
        if not available_nodes:
            for job in jobs:
                assignments[job.job_id] = None
            return assignments

        stock_info = self._get_stock_info()

        meta = []
        for job in jobs:
            e_j = self._min_exec_time(job, available_nodes)
            if e_j == float('inf'):
                meta.append((job, e_j, _NO_NODE))
                continue
            slack = job.deadline - current_time
            cr = slack / e_j if e_j > 0 else float('inf')
            if cr <= 1.0:
                tier = _DOOMED
            elif cr <= self.cr_star:
                tier = _AT_RISK
            else:
                tier = _SAFE
            meta.append((job, e_j, tier))

        def sort_key(m):
            job, e_j, tier = m
            if tier == _AT_RISK:
                return (0, job.deadline)
            if tier == _SAFE:
                return (1, e_j)
            if tier == _DOOMED:
                return (2, job.deadline)
            return (3, job.deadline)

        meta.sort(key=sort_key)

        remaining = list(available_nodes)
        for job, e_j, tier in meta:
            if not remaining or e_j == float('inf'):
                assignments[job.job_id] = None
                continue
            capable = [n for n in remaining if n.can_accept_job(job.required_memory)]
            if not capable:
                assignments[job.job_id] = None
                continue
            node = self._select_node(job, capable, current_time, stock_info)
            if node is not None:
                assignments[job.job_id] = node.node_id
                remaining.remove(node)
            else:
                assignments[job.job_id] = None

        return assignments

    # -- helpers --------------------------------------------------------------

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
        return best

    def _stock_rank(self, node: Node, stock_info: dict) -> int:
        return {StockStatus.HIGH: 0, StockStatus.MEDIUM: 1, StockStatus.LOW: 2}.get(
            stock_info.get(node.resource_type, StockStatus.HIGH), 0)

    def _select_node(self, job: Job, capable: List[Node], current_time: float,
                     stock_info: dict) -> Optional[Node]:
        if not capable:
            return None
        non_low = [n for n in capable
                   if stock_info.get(n.resource_type, StockStatus.HIGH) != StockStatus.LOW]
        pool = non_low if non_low else capable

        if self.cost_aware:
            feasible = []
            for n in pool:
                try:
                    if current_time + self.get_execution_time(job, n) <= job.deadline - self.margin:
                        feasible.append(n)
                except Exception:
                    pass
            if feasible:
                return min(feasible, key=lambda n: (
                    n.get_per_second_cost(),
                    self.get_execution_time(job, n),
                    self._stock_rank(n, stock_info),
                ))

        return min(pool, key=lambda n: (
            self.get_execution_time(job, n),
            n.get_per_second_cost(),
            self._stock_rank(n, stock_info),
        ))

    def get_scheduler_info(self) -> dict:
        return {
            "name": "CADR" if self.cost_aware else "CADR-order-only",
            "description": (
                "Cost-aware deadline-risk: critical-ratio ordering "
                f"(CR*={self.cr_star:.1f}) with "
                + ("cost-first deadline-feasible placement"
                   if self.cost_aware else "fastest-node placement")
            ),
            "cr_star": self.cr_star,
            "margin": self.margin,
            "cost_aware": self.cost_aware,
        }
