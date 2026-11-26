"""
Anticipative Rolling-Horizon Scheduler for CGRSP.

Forward-simulation receding-horizon control. At each scheduling event the
scheduler models every usable slot's availability timeline (idle now, or busy
until the running job's expected completion). Only idle slots receive a
placement (a busy slot's expected free time sets when the next slot frees, which
splits the urgent and normal tiers); the waiting queue is placed onto the idle
slots' timelines, possibly at a later start behind an earlier-ordered placement,
by greedy minimisation of the model bi-objective. Candidates
are ordered in three tiers: urgent jobs that would miss their deadline if they
waited for the next slot to free (sorted earliest-deadline-first), then normal
jobs with slack (sorted shortest-processing-time-first for throughput), then
hopeless jobs whose deadline is already unreachable (sorted last). Only jobs
whose placement starts now on an idle slot are committed; the plan is recomputed
at the next event. The tier orders (SPT, EDF, demote-last) are fixed rules; the
projected slot timeline only sets the boundary between the urgent and normal tiers.

Anticipation (load-gated capacity reservation). The scheduler may hold a small
number of idle slots free for anticipated tight-deadline (rush-order) arrivals,
rather than spend them on loose jobs that have ample slack. Reservation is gated
by the offered load: it is active only when the system is below saturation, where
a held slot would otherwise sit unused for a tight arrival; under saturation the
gate disables reservation entirely, because every slot is needed by the current
queue (a held slot only delays jobs that cannot be deferred). The gate compares
the offered load rho = lambda * mean_service_time / C_max against rho_gate; the
reservation count is reservation_aggressiveness slots (0 disables anticipation).
Tight jobs already in the queue may always use any idle slot; only loose jobs are
withheld from the reserved slots, and never more than C_max - 1 slots are held.
"""
from typing import Dict, List, Optional
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node, NodeState
from cgrsp.data.runpod_availability import StockStatus
from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, GPUType
from cgrsp.data.job_generator import COMPLEXITY_DISTRIBUTION

W_QOS = 0.5
W_COST = 0.5
ALPHA1 = 1.0
ALPHA2 = 10.0
EPS = 1e-6
_PLAN_HORIZON = 7200.0          # internal planning window for slot timelines (not a tunable)
PHANTOM_COMPLEXITY = 2          # representative complexity for the idle-slot probe
TIGHT_WINDOW = 7200.0           # deadline window (s) at or below which a job is a tight (rush) job
_STOCK_PEN = {StockStatus.HIGH: 0.0, StockStatus.MEDIUM: 0.2, StockStatus.LOW: 1.0}
GPU_RELIABILITY_PENALTY = {"RTX_A4000": 0.0, "RTX_A4500": 5.0,
                           "RTX_A5000": 15.0, "RTX_3090": 40.0}


def _mean_service_time() -> float:
    """Expected service time E[W] over the complexity mix, averaged across GPU types.

    Used only for the offered-load gate (rho = lambda * E[W] / C_max); it is a
    fixed property of the calibrated execution-time model, not a tunable.
    """
    gpus = [GPUType.RTX_3090, GPUType.RTX_A5000, GPUType.RTX_A4500, GPUType.RTX_A4000]
    total = 0.0
    for complexity, weight in COMPLEXITY_DISTRIBUTION.items():
        per_gpu = [EXECUTION_TIME_LOOKUP[(complexity, g)] for g in gpus]
        total += weight * (sum(per_gpu) / len(per_gpu))
    return total


_MEAN_SERVICE_TIME = _mean_service_time()


class RollingHorizonScheduler(Scheduler):
    def __init__(self, fleet_manager, seed: Optional[int] = None,
                 reservation_aggressiveness: float = 1.0,
                 arrival_rate: float = 0.1,
                 rho_gate: float = 0.95):
        super().__init__(fleet_manager, seed)
        self.reservation_aggressiveness = reservation_aggressiveness
        self.arrival_rate = arrival_rate
        self.rho_gate = rho_gate
        self._expected_free: Dict[int, float] = {}

    def schedule(self, jobs: List[Job], current_time: float) -> Dict[int, Optional[int]]:
        if not jobs:
            return {}
        idle_ids = {n.node_id for n in self.get_idle_nodes()}
        if not idle_ids:
            return {j.job_id: None for j in jobs}
        slots = self._build_slots(current_time)
        if not slots:
            return {j.job_id: None for j in jobs}

        cands = list(jobs)

        min_e = {}
        for job in cands:
            best = float("inf")
            for s in slots:
                if s["node"].can_accept_job(job.required_memory):
                    e = self.get_execution_time(job, s["node"])
                    if e < best:
                        best = e
            min_e[id(job)] = best

        next_free = self._soonest_free(current_time, slots)
        anticipating = self._reserve_count() > 0

        def key(job):
            e = min_e[id(job)]
            if e == float("inf") or current_time + e > job.deadline:
                return (2, job.deadline, e)      # hopeless -> last
            if anticipating and (job.deadline - job.submit_time) < TIGHT_WINDOW:
                return (0, job.deadline, e)      # anticipating: serve tight jobs eagerly (EDF)
            if next_free + e > job.deadline:
                return (0, job.deadline, e)      # urgent -> EDF
            return (1, e, job.deadline)          # normal -> SPT
        cands.sort(key=key)

        plan_end = current_time + _PLAN_HORIZON
        committed: Dict[int, int] = {}
        used_now = set()
        for job in cands:
            best = None
            best_score = float("inf")
            for s in slots:
                node = s["node"]
                if not node.can_accept_job(job.required_memory):
                    continue
                e = self.get_execution_time(job, node)
                start = self._earliest_start(s["busy"], current_time, e, plan_end)
                if start is None:
                    continue
                wait = max(0.0, start - job.submit_time)
                miss = 1.0 if start + e > job.deadline + EPS else 0.0
                cost = e * node.get_per_second_cost()
                score = (W_QOS * (ALPHA1 * wait + ALPHA2 * miss)
                         + W_COST * cost + self._penalty(node))
                if score < best_score:
                    best_score = score
                    best = (s, start, e, node)
            if best is None:
                continue
            s, start, e, node = best
            s["busy"].append((start, start + e))
            s["busy"].sort()
            if (start <= current_time + EPS
                    and node.node_id in idle_ids and node.node_id not in used_now):
                committed[job.job_id] = node.node_id
                used_now.add(node.node_id)
                self._expected_free[node.node_id] = current_time + e

        self._apply_reservation(committed, jobs, len(idle_ids))
        return {j.job_id: committed.get(j.job_id) for j in jobs}

    def _apply_reservation(self, committed: Dict[int, int], jobs: List[Job],
                           n_idle: int) -> None:
        """Withhold loose-job commits so reserved idle slots stay free for tight arrivals.

        No-op when the offered-load gate yields a zero reservation (e.g. under
        saturation), so the saturated-load behaviour is unchanged.
        """
        reserve = self._reserve_count()
        if reserve <= 0 or not committed:
            return
        by_id = {j.job_id: j for j in jobs}
        tight_used = 0
        loose_order: List[int] = []
        for jid in committed:                       # dict preserves commit (tier) order
            job = by_id[jid]
            if (job.deadline - job.submit_time) < TIGHT_WINDOW:
                tight_used += 1
            else:
                loose_order.append(jid)
        allowed_loose = max(0, n_idle - tight_used - reserve)
        for jid in loose_order[allowed_loose:]:
            del committed[jid]

    def _reserve_count(self) -> int:
        """Number of idle slots to reserve for tight arrivals, gated by offered load."""
        if self.reservation_aggressiveness <= 0.0 or self.arrival_rate <= 0.0:
            return 0
        n_slots = len(self.fleet_manager.get_all_nodes())
        if n_slots <= 1:
            return 0
        rho = self.arrival_rate * _MEAN_SERVICE_TIME / n_slots
        if rho >= self.rho_gate:                     # at/above saturation: never reserve
            return 0
        return min(int(round(self.reservation_aggressiveness)), n_slots - 1)

    def _soonest_free(self, current_time, slots):
        frees = []
        idle_probe = float("inf")
        for s in slots:
            if s["busy"]:
                frees.append(max(s["busy"][-1][1], current_time))
            else:
                idle_probe = min(idle_probe, current_time + self._probe(s["node"]))
        if frees:
            return min(frees)
        return idle_probe if idle_probe < float("inf") else current_time

    def _build_slots(self, current_time: float):
        slots = []
        for n in self.fleet_manager.get_all_nodes():
            st = getattr(n, "state", None)
            if st == NodeState.IDLE:
                slots.append({"node": n, "busy": []})
            elif st == NodeState.RUNNING:
                free = max(self._expected_free.get(n.node_id, current_time + self._probe(n)),
                           current_time)
                slots.append({"node": n, "busy": [(current_time, free)]})
            # UNAVAILABLE nodes are excluded
        return slots

    def _probe(self, node: Node) -> float:
        return self.get_execution_time(
            Job(job_id=-1, complexity=PHANTOM_COMPLEXITY, deadline=0.0,
                submit_time=0.0, estimated_duration=0.0), node)

    @staticmethod
    def _earliest_start(busy, avail, e, plan_end):
        start = avail
        for (bs, be) in busy:            # busy kept sorted ascending, non-overlapping
            if be <= start:
                continue
            if bs >= start + e:
                break
            start = be
        return start if start <= plan_end + EPS else None

    def _penalty(self, node: Node) -> float:
        status = StockStatus.HIGH
        if hasattr(self.fleet_manager, "get_stock_status"):
            status = self.fleet_manager.get_stock_status(node.resource_type)
        return (_STOCK_PEN.get(status, 0.0)
                + GPU_RELIABILITY_PENALTY.get(node.resource_type.value, 20.0) / 100.0)

    def get_scheduler_info(self) -> dict:
        return {"name": "RH",
                "reservation_aggressiveness": self.reservation_aggressiveness,
                "rho_gate": self.rho_gate}
