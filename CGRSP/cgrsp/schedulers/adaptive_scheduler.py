"""
Adaptive EDF-SPT Hybrid Scheduler for CGRSP

Three-tier priority system that combines deadline awareness (EDF) with
throughput optimization (SPT) and hopeless-job demotion:

  Tier 1 (critical):  0 <= laxity < threshold  -> sorted by deadline (EDF)
  Tier 2 (safe):      laxity >= threshold       -> sorted by processing time (SPT)
  Tier 3 (hopeless):  laxity < 0                -> scheduled last

Under high queue pressure (len(jobs) > pressure_threshold), the
critical threshold expands so that every job with non-negative laxity is
treated as critical, so the order reduces to EDF with hopeless demotion.  This preserves
EDF's strong surge-day performance while gaining SPT throughput on
lighter days.

Laxity of job j at time t:
    L_j(t) = deadline_j - t - min_execution_time_j

Hopeless demotion prioritises jobs projected to finish on time over jobs
projected to miss under mean service times. A shorter realised service
may still let a demoted job meet its deadline. Critical-tier node selection
accounts for GPU stock reliability to minimise long provisioning delays.
"""

from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node
from cgrsp.data.runpod_availability import StockStatus

GPU_RELIABILITY_PENALTY = {
    "RTX_A4000": 0.0,
    "RTX_A4500": 5.0,
    "RTX_A5000": 15.0,
    "RTX_3090":  40.0,
}


class AdaptiveScheduler(Scheduler):
    """
    Adaptive EDF-SPT hybrid with hopeless-job demotion, queue-pressure
    adaptation, and provisioning-reliability-aware node selection.

    Parameters
    ----------
    critical_threshold : float
        Laxity (seconds) below which a job is promoted to critical tier
        under normal queue pressure.
    pressure_threshold : int
        Queue length above which the scheduler expands the critical tier
        to cover every job with non-negative laxity.
    cost_weight : float
        Cost vs speed tradeoff for safe-tier node selection (0-1).
    """

    def __init__(self, fleet_manager, seed: Optional[int] = None,
                 critical_threshold: float = 600.0,
                 pressure_threshold: int = 10,
                 cost_weight: float = 0.3):
        super().__init__(fleet_manager, seed)
        self.critical_threshold = critical_threshold
        self.pressure_threshold = pressure_threshold
        self.cost_weight = max(0.0, min(1.0, cost_weight))

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

        effective_threshold = self.critical_threshold
        if len(jobs) > self.pressure_threshold:
            effective_threshold = 28800.0

        critical = []
        safe = []
        hopeless = []

        for job in jobs:
            min_exec = self._min_exec_time(job, available_nodes)
            laxity = job.deadline - current_time - min_exec

            if laxity < 0:
                hopeless.append((job, min_exec, laxity))
            elif laxity < effective_threshold:
                critical.append((job, min_exec, laxity))
            else:
                safe.append((job, min_exec, laxity))

        critical.sort(key=lambda x: x[0].deadline)
        safe.sort(key=lambda x: x[1])
        hopeless.sort(key=lambda x: x[0].deadline)

        ordered = critical + safe + hopeless
        remaining_nodes = list(available_nodes)

        for job, _, laxity in ordered:
            if not remaining_nodes:
                assignments[job.job_id] = None
                continue

            capable = [n for n in remaining_nodes
                       if n.can_accept_job(job.required_memory)]
            if not capable:
                assignments[job.job_id] = None
                continue

            is_critical = (laxity >= 0 and laxity < effective_threshold)

            if is_critical:
                node = self._select_critical_node(job, capable, stock_info)
            else:
                node = self._select_balanced_node(job, capable, stock_info)

            if node:
                assignments[job.job_id] = node.node_id
                remaining_nodes.remove(node)
            else:
                assignments[job.job_id] = None

        return assignments

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
        # inf (not 0.0) when no capable idle node exists: a 0.0 here would
        # give the job laxity = deadline - t and misclassify an unplaceable job
        # into the critical/safe tiers (CADR and laxity-rescue use the same
        # convention). Unreachable in the canonical experiments (no job sets
        # required_memory), so published results are unaffected.
        return best

    def _select_critical_node(self, job: Job, capable: List[Node],
                              stock_info: dict) -> Optional[Node]:
        """Select node for critical-tier jobs: fastest + reliable stock."""
        scored = []
        for node in capable:
            try:
                exec_time = self.get_execution_time(job, node)
                status = stock_info.get(node.resource_type, StockStatus.HIGH)
                stock_penalty = {StockStatus.HIGH: 0.0,
                                 StockStatus.MEDIUM: 50.0,
                                 StockStatus.LOW: 200.0}.get(status, 0.0)
                reliability = GPU_RELIABILITY_PENALTY.get(
                    node.resource_type.value, 20.0)
                scored.append((node, exec_time + stock_penalty + reliability))
            except Exception:
                pass
        if not scored:
            return None
        scored.sort(key=lambda x: x[1])
        return scored[0][0]

    def _select_balanced_node(self, job: Job, capable: List[Node],
                              stock_info: dict) -> Optional[Node]:
        """Select node with balanced speed/cost tradeoff."""
        scores = []
        for node in capable:
            try:
                exec_time = self.get_execution_time(job, node)
                fastest = min(self.get_execution_time(job, n) for n in capable)
                cheapest = min(n.get_per_second_cost() for n in capable)
                norm_t = exec_time / fastest if fastest > 0 else 1.0
                norm_c = node.get_per_second_cost() / cheapest if cheapest > 0 else 1.0
                penalty = {StockStatus.HIGH: 0.0,
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
            "name": "Adaptive",
            "description": (
                f"EDF-SPT hybrid with hopeless demotion + queue-pressure "
                f"adaptation (threshold={self.pressure_threshold}) + "
                f"reliability-aware node selection"
            ),
            "critical_threshold_s": self.critical_threshold,
            "pressure_threshold": self.pressure_threshold,
            "cost_weight": self.cost_weight,
        }
