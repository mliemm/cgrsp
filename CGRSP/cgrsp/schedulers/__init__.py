"""
CGRSP Scheduling Algorithms

Provides scheduling algorithms for the Cloud GPU Rendering
Scheduling Problem (CGRSP). All schedulers implement the base Scheduler interface
and can be used interchangeably with the simulator.

Available schedulers:
- FIFOScheduler: First-In-First-Out (processes jobs in submission order)
- SPTScheduler: Shortest Processing Time (maximizes throughput)
- EDFScheduler: Earliest Deadline First (prioritizes urgent jobs)
- LCFScheduler: Lowest Cost First (prioritizes cheap resources)
- BalancedScheduler: Weighted trade-off between speed and cost
- RandomScheduler: Baseline random assignment
- LaxityRescueScheduler: SPT with EDF rescue below laxity threshold
- RollingHorizonScheduler: Anticipative rolling-horizon planning
- CostAwareDeadlineRiskScheduler: Critical-ratio priority with cheapest-feasible node selection
"""

from cgrsp.schedulers.base import Scheduler, SchedulerMetrics
from cgrsp.schedulers.fifo_scheduler import (
    FIFOScheduler,
    create_fifo_scheduler,
)
from cgrsp.schedulers.spt_scheduler import (
    SPTScheduler,
    create_spt_scheduler,
)
from cgrsp.schedulers.edf_scheduler import (
    EDFScheduler,
    create_edf_scheduler,
)
from cgrsp.schedulers.lcf_scheduler import (
    LCFScheduler,
    create_lcf_scheduler,
)
from cgrsp.schedulers.balanced_scheduler import (
    BalancedScheduler,
    create_balanced_scheduler,
)
from cgrsp.schedulers.random_scheduler import (
    RandomScheduler,
    create_random_scheduler,
)
from cgrsp.schedulers.laxity_rescue_scheduler import (
    LaxityRescueScheduler,
)
from cgrsp.schedulers.rolling_horizon_scheduler import (
    RollingHorizonScheduler,
)
from cgrsp.schedulers.cost_aware_deadline_risk import (
    CostAwareDeadlineRiskScheduler,
)
from cgrsp.schedulers.adaptive_scheduler import (
    AdaptiveScheduler,
)

__all__ = [
    "Scheduler",
    "SchedulerMetrics",
    "FIFOScheduler",
    "create_fifo_scheduler",
    "SPTScheduler",
    "create_spt_scheduler",
    "EDFScheduler",
    "create_edf_scheduler",
    "LCFScheduler",
    "create_lcf_scheduler",
    "BalancedScheduler",
    "create_balanced_scheduler",
    "RandomScheduler",
    "create_random_scheduler",
    "LaxityRescueScheduler",
    "RollingHorizonScheduler",
    "CostAwareDeadlineRiskScheduler",
    "AdaptiveScheduler",
]
