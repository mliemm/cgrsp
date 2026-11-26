"""
Shallow tabular contextual bandit baseline for CGRSP.

This is a four-armed contextual bandit over job orderings. It is not a
reinforcement-learning agent and not a DQN: there is no transition model, no
value function, no bootstrapping, no gradient and no neural network. The whole
policy is a table of four numbers per context, frozen before evaluation and
read with epsilon = 0.

Arms (the sequence in which queued jobs are offered an idle slot):

    0  fifo        submit_time ascending, then job id
    1  spt         minimum execution time over the idle slots ascending
    2  edf         deadline ascending, then job id
    3  spt_rescue  SPT+Rescue ordering: a job whose laxity
                   d_j - t - e(j, g) falls below 600 s joins an urgent tier
                   ordered by deadline; the rest keep SPT order

Every arm shares one node rule, the idle slot with the earliest finish time
current_time + W(kappa_j, g), ties to the lower slot id. The arm therefore
changes only the ordering, so a difference between arms isolates sequencing
from placement.

Context, from fixed constants declared before any result was inspected:

    queue bucket   len(queued jobs) < 10 / 10..49 / >= 50
    tight bucket   share of queued jobs whose deadline window
                   d_j - t_j^submit is at most 1 h: < 1/6 / 1/6..1/3 / >= 1/3
    stock bucket   number of supported GPU types currently LOW: 0 / 1 / >= 2

The buckets form at most 27 contexts, keyed as "q{0..2}|t{0..2}|s{0..2}". An
unseen context reads as all-equal values and falls back to arm 0 (FIFO order).
An arm that training never drew in an otherwise visited context keeps its 0.0
initial value; because every observed episode reward is negative, that untried
arm outranks the tried ones and is chosen first (optimistic initial values).

Reward, episode level and assigned post hoc:

    r = -miss_rate - 1e-5 * mean_wait_min

miss_rate is the share of dispatched jobs whose realised completion time
exceeded the deadline, and mean_wait_min is the mean of
start_time - submit_time over the same dispatched jobs. The 1e-5 coefficient
makes the wait term a tiebreak only: 100 minutes of mean wait move the reward
as much as 0.1 percentage points of miss rate. The table never sees a reward
while it is making a decision. An episode runs to completion, episode_reward()
then reads the dispatch log and the recorded completions, and every
(context, arm) pair visited during that episode receives the same episode
reward, once per pair (update_q_table). That is a deliberately coarse credit
assignment: a context visited in a bad episode is blamed along with every other
decision in the same episode. It is kept deliberately, because this baseline
must stay shallow and its results must not be overstated.

Frozen protocol:

    training    seeds 500-514 (15 seeds), epsilon = 0.25, incremental-mean updates
    evaluation  seeds 0-29, epsilon = 0, frozen table

The training range was halved from the 30 seeds first sketched to 15 after the
30-seed run measured 5 min 27 s in total and the experiment has a 5-minute
runtime target. The reduction removes training episodes only; it was not chosen for any
result. The two ranges are asserted disjoint at import (assert_disjoint_seeds).
"Bandit-random" in the experiment is the same class with an empty table and
epsilon = 1.0, that is a uniform random arm at every scheduling call; it is the
control that separates learning from the mere diversity of the four orderings.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

from cgrsp.config import ResourceType
from cgrsp.data.runpod_availability import StockStatus
from cgrsp.schedulers.base import Scheduler
from cgrsp.simulator.queue import Job

ARMS: Tuple[str, ...] = ("fifo", "spt", "edf", "spt_rescue")

# Context buckets. Fixed constants, not fitted to any outcome.
QUEUE_LENGTH_BOUNDS: Tuple[int, ...] = (10, 50)
TIGHT_FRACTION_BOUNDS: Tuple[float, ...] = (1.0 / 6.0, 1.0 / 3.0)
LOW_STOCK_BOUNDS: Tuple[int, ...] = (1, 2)
TIGHT_WINDOW_SEC: float = 3600.0

# SPT+Rescue promotion threshold, same 600 s default as LaxityRescueScheduler.
RESCUE_THRESHOLD_SEC: float = 600.0

# Reward: negative deadline-miss rate, with a wait tiebreak.
WAIT_TIEBREAK_PER_MIN: float = 1e-5

# Frozen seed protocol.
TRAIN_SEEDS: Tuple[int, ...] = tuple(range(500, 515))
EVAL_SEEDS: Tuple[int, ...] = tuple(range(0, 30))


def assert_disjoint_seeds(
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
) -> None:
    """Raise if any bandit training seed is also an evaluation seed."""
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise ValueError(
            "Bandit training and evaluation seeds must be disjoint; "
            f"overlap: {overlap}"
        )


assert_disjoint_seeds()


def _bucket(value: float, bounds: Sequence[float]) -> int:
    """Map a value to 0..len(bounds) by the declared ascending boundaries."""
    index = 0
    for bound in bounds:
        if value >= bound:
            index += 1
    return index


def context_key(
    queue_length: int,
    tight_fraction: float,
    low_stock_count: int,
) -> str:
    """Canonical string key for one context, JSON-safe for the frozen table."""
    return "q{}|t{}|s{}".format(
        _bucket(queue_length, QUEUE_LENGTH_BOUNDS),
        _bucket(tight_fraction, TIGHT_FRACTION_BOUNDS),
        _bucket(low_stock_count, LOW_STOCK_BOUNDS),
    )


class BanditScheduler(Scheduler):
    """
    Four-armed tabular contextual bandit over job orderings.

    Parameters
    ----------
    fleet_manager : FleetManager
        Provides idle slots and qualitative stock status.
    seed : int, optional
        Seed of the private exploration stream. The stream is independent of
        the simulator's RNG, so exploration never perturbs fleet or execution
        sampling.
    q_table : dict, optional
        Frozen table mapping a context key to four arm values. It is copied on
        construction, so later mutations of the caller's dict are not observed.
        None means an empty table (all values equal, arm 0).
    epsilon : float
        Exploration probability in [0, 1]. Evaluation uses 0.0; the
        Bandit-random control uses 1.0.
    """

    def __init__(
        self,
        fleet_manager,
        seed: Optional[int] = None,
        q_table: Optional[Dict[str, Sequence[float]]] = None,
        epsilon: float = 0.0,
    ):
        super().__init__(fleet_manager, seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")

        table: Dict[str, List[float]] = {}
        for key, values in (q_table or {}).items():
            if len(values) != len(ARMS):
                raise ValueError(
                    f"q_table entry {key!r} has {len(values)} values, "
                    f"expected {len(ARMS)}"
                )
            table[key] = [float(value) for value in values]

        self.q_table = table
        self.epsilon = float(epsilon)
        # Execution times are deterministic per (complexity, resource type) and
        # are asked for on every call; memoise them for the run.
        self._exec_cache: Dict[tuple, float] = {}
        self._rng = random.Random(0 if seed is None else seed)
        # One (context, arm) entry per scheduling call, used only by the
        # post-hoc training update. Empty at evaluation.
        self.decision_log: List[Tuple[str, int]] = []

    def _exec(self, job: Job, node) -> float:
        """Memoised self.get_execution_time(job, node)."""
        key = (int(job.complexity), node.resource_type)
        cached = self._exec_cache.get(key)
        if cached is None:
            cached = self.get_execution_time(job, node)
            self._exec_cache[key] = cached
        return cached

    # Context

    def _low_stock_count(self) -> int:
        """Number of supported GPU types whose current stock status is LOW."""
        getter = getattr(self.fleet_manager, "get_stock_status", None)
        if getter is None:
            return 0
        return sum(
            1 for resource_type in ResourceType.supported_gpus()
            if getter(resource_type) == StockStatus.LOW
        )

    def context(self, jobs: Sequence[Job], current_time: Optional[float] = None) -> str:
        """Context key for the queue as observed at this scheduling call."""
        if jobs:
            tight = sum(
                1 for job in jobs
                if job.deadline - job.submit_time <= TIGHT_WINDOW_SEC
            )
            tight_fraction = tight / len(jobs)
        else:
            tight_fraction = 0.0
        return context_key(len(jobs), tight_fraction, self._low_stock_count())

    # Arm choice

    def _select_arm(self, key: str) -> int:
        """Epsilon-greedy read of the frozen table (ties: lowest arm index)."""
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.randrange(len(ARMS))

        values = self.q_table.get(key)
        if values is None:
            return 0
        best = 0
        for index in range(1, len(ARMS)):
            if values[index] > values[best]:
                best = index
        return best

    # Ordering

    def _min_exec_by_job(
        self,
        jobs: Sequence[Job],
        idle_nodes: Sequence,
    ) -> Dict[int, Optional[float]]:
        """Fastest execution time of each job over the idle slots, None if none fit."""
        fastest: Dict[int, Optional[float]] = {}
        for job in jobs:
            best: Optional[float] = None
            for node in idle_nodes:
                if not node.can_accept_job(job.required_memory):
                    continue
                execution = self._exec(job, node)
                if best is None or execution < best:
                    best = execution
            fastest[job.job_id] = best
        return fastest

    def _order_jobs(
        self,
        arm: str,
        jobs: Sequence[Job],
        current_time: float,
        idle_nodes: Sequence,
    ) -> List[Job]:
        """Return the queued jobs in the order selected by one arm."""
        if arm == "fifo":
            return sorted(jobs, key=lambda job: (job.submit_time, job.job_id))
        if arm == "edf":
            return sorted(jobs, key=lambda job: (job.deadline, job.job_id))

        fastest = self._min_exec_by_job(jobs, idle_nodes)

        if arm == "spt":
            def spt_key(job: Job):
                execution = fastest[job.job_id]
                return (execution is None, execution or 0.0, job.job_id)
            return sorted(jobs, key=spt_key)

        if arm == "spt_rescue":
            def rescue_key(job: Job):
                execution = fastest[job.job_id]
                if execution is None:
                    # No idle slot can take it this round; behind both tiers.
                    return (2, 0.0, job.job_id)
                laxity = job.deadline - current_time - execution
                if laxity < RESCUE_THRESHOLD_SEC:
                    return (0, job.deadline, job.job_id)
                return (1, execution, job.job_id)
            return sorted(jobs, key=rescue_key)

        raise ValueError(f"Unknown bandit arm: {arm!r}")

    # Scheduling

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """Pick an arm for this context, then dispatch with its ordering."""
        assignments: Dict[int, Optional[int]] = {job.job_id: None for job in jobs}
        if not jobs:
            return assignments

        key = self.context(jobs, current_time)
        arm_index = self._select_arm(key)
        self.decision_log.append((key, arm_index))

        available = sorted(self.get_idle_nodes(), key=lambda node: node.node_id)
        if not available:
            return assignments

        ordered = self._order_jobs(ARMS[arm_index], jobs, current_time, available)

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


def episode_reward(dispatch_log: Sequence[tuple], get_job_metrics) -> float:
    """
    Episode-level reward from the dispatch log and the recorded completions.

    Parameters
    ----------
    dispatch_log : sequence of (job_id, node_id, dispatch_time, start_time)
        The simulator's dispatch log for the episode.
    get_job_metrics : callable
        Metrics-collector accessor returning one job's record, whose
        completion_time and deadline define the miss.

    Returns
    -------
    float
        -miss_rate - 1e-5 * mean_wait_min over the dispatched jobs. Jobs the
        episode never dispatched are excluded; the shared 24 h window truncates
        the tail for every policy alike.
    """
    misses = 0
    total_wait = 0.0
    counted = 0
    for job_id, _node_id, _dispatch_time, start_time in dispatch_log:
        metrics = get_job_metrics(job_id)
        if metrics is None or metrics.completion_time is None:
            continue
        counted += 1
        if metrics.completion_time > metrics.deadline:
            misses += 1
        total_wait += max(0.0, start_time - metrics.submit_time)

    if counted == 0:
        return 0.0

    miss_rate = misses / counted
    mean_wait_min = total_wait / counted / 60.0
    return -miss_rate - WAIT_TIEBREAK_PER_MIN * mean_wait_min


def update_q_table(
    q_table: Dict[str, List[float]],
    counts: Dict[str, List[int]],
    visited_pairs: Sequence[Tuple[str, int]],
    reward: float,
) -> None:
    """
    Incremental-mean update, one credit per (context, arm) pair per episode.

    Repeated pairs in ``visited_pairs`` are credited once, so a context visited
    a hundred times in one episode is not mistaken for a hundred observations.
    Unseen keys are initialised to zero.
    """
    for key, arm in dict.fromkeys(visited_pairs):
        row = q_table.setdefault(key, [0.0] * len(ARMS))
        row_counts = counts.setdefault(key, [0] * len(ARMS))
        row_counts[arm] += 1
        row[arm] += (reward - row[arm]) / row_counts[arm]


def create_bandit_scheduler(fleet_manager, seed: Optional[int] = None) -> BanditScheduler:
    """Factory function to create a BanditScheduler (empty table, greedy)."""
    return BanditScheduler(fleet_manager, seed)
