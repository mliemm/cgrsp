"""
Max-Min baseline for CGRSP.

Max-Min is the counterpart of Min-Min (min_min_scheduler.py): each round every
unscheduled job is matched with the idle node on which it would finish
earliest, and then the job with the largest of those earliest completion times
is assigned first. It therefore drains the long jobs while the short ones are
still cheap to place later, the opposite bias to Min-Min.

Only the selection polarity differs, so the procedure, the tie-breaking
(smallest completion time first for Min-Min, largest first for Max-Min, then
lower job id, then lower node id) and the VRAM filtering are inherited from
MinMinScheduler.
"""

from typing import Dict, List, Optional

from cgrsp.schedulers.min_min_scheduler import MinMinScheduler
from cgrsp.simulator.queue import Job


class MaxMinScheduler(MinMinScheduler):
    """
    Max-Min: repeatedly assign the job with the largest earliest completion time.

    Inherits node filtering and tie-breaking from MinMinScheduler; only
    ``pick_largest`` is overridden.
    """

    pick_largest: bool = True

    def schedule(
        self,
        jobs: List[Job],
        current_time: float
    ) -> Dict[int, Optional[int]]:
        """Assign queued jobs by repeated largest-earliest-completion selection."""
        return super().schedule(jobs, current_time)


def create_max_min_scheduler(fleet_manager, seed: Optional[int] = None) -> MaxMinScheduler:
    """Factory function to create a MaxMinScheduler."""
    return MaxMinScheduler(fleet_manager, seed)
