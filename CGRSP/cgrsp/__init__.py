"""
CGRSP: Cloud GPU Rendering Scheduling Problem Simulator

A discrete-event simulation framework for GPU rendering job scheduling
with stochastic availability and deadline constraints.

**GPU Fleet Configuration:**
CGRSP supports exactly 4 GPU types with empirical performance data:
- RTX_A5000: fastest on the pooled dataset (1.00x speed, $0.27/hr)
- RTX_3090: second (0.95x speed, $0.46/hr; fastest at low complexity)
- RTX_A4500: third (0.86x speed, $0.25/hr; fastest at high complexity)
- RTX_A4000: slowest (0.76x speed, $0.25/hr)
(relative speeds from config.py base_speed; the ranking inverts by complexity class,
see execution_times.py)

All models are based on 6,234 real rendering jobs from the author's RunPod rendering
pipeline (6,030 production, 203 development-tagged, 1 untagged).
No other GPU types are supported.
"""

__version__ = "0.1.0"
__author__ = "Thesis Project"

from cgrsp.simulator.core import Simulator, Event, EventType
from cgrsp.simulator.fleet import Node, NodeState, FleetManager
from cgrsp.simulator.queue import Job, JobQueue
from cgrsp.simulator.state import ObservableState
from cgrsp.simulator.metrics import MetricsCollector, JobMetrics

__all__ = [
    "Simulator",
    "Event",
    "EventType",
    "Node",
    "NodeState",
    "FleetManager",
    "Job",
    "JobQueue",
    "ObservableState",
    "MetricsCollector",
    "JobMetrics",
]
