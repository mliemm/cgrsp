"""
Core Simulator Engine Module

Implements discrete-event simulation with continuous time representation
for the CGRSP scheduling problem. The event loop is adapted from SPARS
(Amrizal et al., SoftwareX 34:102693, 2026).
"""

import heapq
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum

from cgrsp.config import ResourceType, FleetConfig, TemporalParameters, AvailabilityParameters
from cgrsp.simulator.fleet import FleetManager, NodeState
from cgrsp.simulator.queue import Job, JobQueue, JobState
from cgrsp.simulator.state import ObservableState, NodeStatus
from cgrsp.simulator.metrics import MetricsCollector


class EventType(str, Enum):
    """Types of events in the simulation."""
    JOB_ARRIVAL = "job_arrival"
    JOB_COMPLETION = "job_completion"
    NODE_TRANSITION = "node_transition"  # Stochastic availability change
    SIMULATION_END = "simulation_end"
    SCHEDULER_DECISION = "scheduler_decision"  # Trigger scheduling logic


@dataclass
class Event:
    """
    Represents a discrete event in the simulation.

    Events are ordered by timestamp in the event queue.
    """

    event_type: EventType
    timestamp: float
    priority: int = 0  # For tie-breaking (higher = earlier)
    data: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Event") -> bool:
        """
        Compare events for priority queue ordering.

        Earlier timestamps have higher priority. For equal timestamps,
        higher priority values are processed first.
        """
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority > other.priority


class Simulator:
    """
    Discrete-event simulator for the CGRSP problem.

    **GPU Fleet Support:**
    CGRSP simulator supports exactly 4 GPU types with empirical data:
    - RTX_A5000 (fastest on the pooled dataset, 24GB VRAM, $0.27/hr)
    - RTX_3090 (second fastest overall, 24GB VRAM, $0.46/hr)
    - RTX_A4500 (third, 20GB VRAM, $0.25/hr)
    - RTX_A4000 (slowest, 16GB VRAM, $0.25/hr)
    Speed ranks follow config.py base_speed and invert by complexity class.

    All execution times and pricing are derived from 6,234 real rendering jobs.
    No other GPU types are supported.

    Features:
    - Continuous time representation
    - Event queue with priority handling
    - Stochastic node availability (Markov process)
    - Job queue with deadline-based priority
    - Metrics collection and tracking
    - Extensible scheduler interface
    """

    def __init__(
        self,
        fleet_config: Optional[List[tuple]] = None,
        temporal_params: Optional[TemporalParameters] = None,
        availability_params: Optional[AvailabilityParameters] = None,
        random_seed: Optional[int] = None,
        fleet_manager: Optional[FleetManager] = None,
    ):
        """
        Initialize the simulator.

        Args:
            fleet_config: List of (ResourceType, count) tuples. Optional if fleet_manager provided.
            temporal_params: Temporal parameters (uses defaults if None)
            availability_params: Availability parameters (uses defaults if None)
            random_seed: Random seed for reproducibility
            fleet_manager: Custom fleet manager instance (optional)
        """
        if random_seed is not None:
            random.seed(random_seed)

        # Dedicated RNG for stochastic execution-time sampling. Isolated from the
        # global `random` stream so that node-availability transitions remain
        # identical to deterministic runs (preserving cross-policy comparability).
        self._exec_rng = random.Random(random_seed)

        # Configuration
        self.temporal_params = temporal_params or TemporalParameters()
        self.availability_params = availability_params or AvailabilityParameters()

        # Core components
        if fleet_manager:
            self.fleet_manager = fleet_manager
        elif fleet_config:
            self.fleet_manager = FleetManager(
                fleet_config=fleet_config,
                availability_params=self.availability_params,
                random_seed=random_seed,
            )
        else:
            raise ValueError("Either fleet_config or fleet_manager must be provided")

        self.job_queue = JobQueue()
        self.metrics_collector = MetricsCollector()

        # State tracking
        self.current_time: float = 0.0
        self.observable_state = ObservableState()
        self._next_job_id: int = 0

        # Event queue
        self._event_queue: List[Event] = []
        self._event_counter: int = 0  # For unique event ordering

        # Simulation control
        self.running: bool = False
        self.scheduler: Optional[Callable] = None
        self.scheduler_object: Optional[object] = None  # Store scheduler object

        # Initialize observable state
        self._update_observable_state()

    def _update_observable_state(self) -> None:
        """Update the observable state from fleet and queue."""
        # Update current time
        self.observable_state.current_time = self.current_time

        # Calculate hour of day
        self.observable_state.hour_of_day = int(self.current_time / 3600) % 24

        # Update node statuses
        for node in self.fleet_manager.get_all_nodes():
            self.observable_state.node_statuses[node.node_id] = NodeStatus(
                node_id=node.node_id,
                resource_type=node.resource_type,
                state=node.state.value,
                current_job_id=node.current_job_id,
                available_since=node.state_entry_time,
                unavailable_reason=node.unavailable_reason.value if node.unavailable_reason else None,
            )

        # Update qualitative stock statuses (High/Low) - No exact counts
        if hasattr(self.fleet_manager, 'get_stock_status'):
            for rt in ResourceType.supported_gpus():
                self.observable_state.stock_statuses[rt] = self.fleet_manager.get_stock_status(rt).value

        # Update queued jobs
        self.observable_state.queued_jobs = [
            job.job_id for job in self.job_queue.get_all_jobs()
        ]

        # Update statistics
        self.observable_state.total_jobs_submitted = self.metrics_collector.system_metrics.total_jobs_submitted
        self.observable_state.total_jobs_completed = self.metrics_collector.system_metrics.total_jobs_completed
        self.observable_state.total_jobs_deadline_missed = self.metrics_collector.system_metrics.total_jobs_deadline_missed

    def schedule_event(
        self,
        event_type: EventType,
        timestamp: float,
        data: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> None:
        """
        Schedule an event for future processing.

        Args:
            event_type: Type of event
            timestamp: When the event occurs (must be >= current_time)
            data: Event-specific data
            priority: Priority for tie-breaking (higher = earlier)
        """
        if timestamp < self.current_time:
            raise ValueError(f"Event timestamp ({timestamp}) cannot be in the past (current: {self.current_time})")

        event = Event(
            event_type=event_type,
            timestamp=timestamp,
            priority=priority,
            data=data or {},
        )

        heapq.heappush(self._event_queue, event)

    def _pop_next_event(self) -> Optional[Event]:
        """
        Get the next event from the queue.

        Returns:
            Next event, or None if queue is empty
        """
        if not self._event_queue:
            return None

        return heapq.heappop(self._event_queue)

    def get_next_event_time(self) -> Optional[float]:
        """
        Get the timestamp of the next scheduled event.

        Returns:
            Next event timestamp, or None if no events scheduled
        """
        if not self._event_queue:
            return None

        return self._event_queue[0].timestamp

    def submit_job(
        self,
        complexity: float,
        deadline: float,
        estimated_duration: float,
        required_memory: Optional[float] = None,
        preferred_resource: Optional[ResourceType] = None
    ) -> Job:
        """
        Submit a new job to the system.

        Args:
            complexity: Job complexity κ_j
            deadline: Job deadline (absolute time)
            estimated_duration: Estimated execution duration
            required_memory: Optional VRAM requirement
            preferred_resource: Preferred GPU type

        Returns:
            Created Job object
        """
        job_id = self._next_job_id
        self._next_job_id += 1

        job = Job(
            job_id=job_id,
            complexity=complexity,
            deadline=deadline,
            submit_time=self.current_time,
            estimated_duration=estimated_duration,
            required_memory=required_memory,
            preferred_resource=preferred_resource.value if preferred_resource else None,
        )

        # Add to queue
        self.job_queue.push(job)

        # Record metrics
        self.metrics_collector.record_job_submission(job_id, self.current_time, deadline)

        # Update observable state
        self._update_observable_state()

        # Trigger scheduler decision for new job
        self.schedule_event(
            EventType.SCHEDULER_DECISION,
            self.current_time,
            {"triggered_by_job": job_id},
            priority=10,
        )

        return job

    def _handle_job_arrival(self, event: Event) -> None:
        """
        Handle a job arrival event.

        This creates a new job and adds it to the queue.
        """
        job_data = event.data

        job = self.submit_job(
            complexity=job_data.get("complexity", 1.0),
            deadline=job_data.get("deadline", self.current_time + self.temporal_params.T_deadline),
            estimated_duration=job_data.get("estimated_duration", 3600.0),
            required_memory=job_data.get("required_memory"),
            preferred_resource=job_data.get("preferred_resource"),
        )

        # Trigger scheduler decision
        self.schedule_event(
            EventType.SCHEDULER_DECISION,
            self.current_time,
            {"triggered_by_job": job.job_id},
            priority=10,
        )

    def _handle_job_completion(self, event: Event) -> None:
        """
        Handle a job completion event.

        This frees up the node and records metrics.
        """
        node_id = event.data["node_id"]
        job_id = event.data["job_id"]

        # Get node and calculate cost BEFORE completing the job
        # Use job_start_time to calculate actual execution time, or state_entry_time as fallback
        node = self.fleet_manager.get_node(node_id)
        execution_cost = 0.0
        if node:
            # Try to use job_start_time first (set when job started), fall back to state_entry_time
            start_time = node.job_start_time if hasattr(node, 'job_start_time') and node.job_start_time >= 0 else node.state_entry_time
            if start_time > 0 or start_time == 0:  # Allow start_time of 0 (jobs can start at t=0)
                seconds_used = self.current_time - start_time
                execution_cost = seconds_used * node.get_per_second_cost()

        # Complete job on node
        completed_job_id = self.fleet_manager.complete_job_on_node(node_id, self.current_time)

        if completed_job_id is not None:
            # Record completion metrics
            self.metrics_collector.record_job_completion(
                job_id=completed_job_id,
                completion_time=self.current_time,
                execution_cost=execution_cost,
            )

        # Update state and trigger scheduler
        self._update_observable_state()

        self.schedule_event(
            EventType.SCHEDULER_DECISION,
            self.current_time,
            {"triggered_by_completion": True},
            priority=10,
        )

    def _handle_node_transition(self, event: Event) -> None:
        """
        Handle a stochastic node availability transition.

        This executes a Markov state transition on a node.
        """
        node_id = event.data["node_id"]

        # Execute the transition
        transition_occurred = self.fleet_manager.execute_stochastic_transition(
            node_id=node_id,
            current_time=self.current_time,
        )

        if transition_occurred:
            node = self.fleet_manager.get_node(node_id)

            # If node became unavailable while running a job, handle job interruption
            if node and node.state == NodeState.UNAVAILABLE:
                if node.current_job_id is not None:
                    # In a real system, we might reschedule the job
                    # For now, we just note it was interrupted
                    pass

            # If node became available, trigger scheduler
            if node and node.state == NodeState.IDLE:
                self.schedule_event(
                    EventType.SCHEDULER_DECISION,
                    self.current_time,
                    {"triggered_by_availability": True},
                    priority=10,
                )

            # Schedule next transition for this node
            time_to_next = self.fleet_manager.calculate_time_to_next_transition(
                node_id=node_id,
                current_time=self.current_time,
            )

            if time_to_next < float('inf'):
                self.schedule_event(
                    EventType.NODE_TRANSITION,
                    self.current_time + time_to_next,
                    {"node_id": node_id},
                )

        self._update_observable_state()

    def _handle_scheduler_decision(self, event: Event) -> None:
        """
        Handle a scheduler decision event.

        This invokes the registered scheduler to make assignment decisions.
        """
        if self.scheduler is not None or self.scheduler_object is not None:
            # Use custom scheduler (either function or object)
            self._custom_scheduler()
        else:
            # Default simple FIFO scheduler
            self._default_scheduler()

        self._update_observable_state()

    def _custom_scheduler(self) -> None:
        """
        Custom scheduler handler.

        Gets jobs from queue, calls scheduler, then processes assignments.
        """
        # Get queued jobs
        queued_jobs = self.job_queue.get_jobs_sorted_by_priority()

        if not queued_jobs:
            return

        # Call scheduler to get assignments
        if self.scheduler_object and hasattr(self.scheduler_object, 'schedule'):
            assignments = self.scheduler_object.schedule(queued_jobs, self.current_time)
        elif self.scheduler:
            # Backward compatibility with function-based schedulers
            self.scheduler(self.observable_state, self)
            return
        else:
            return

        self._dispatch_assignments(queued_jobs, assignments)

    def _dispatch_assignments(self, queued_jobs, assignments) -> None:
        """Execute placements through the shared provisioning and service model."""
        # Process assignments
        for job in queued_jobs:
            node_id = assignments.get(job.job_id)

            if node_id is None:
                # Job waits, keep in queue
                continue

            # Assign job to node
            node = self.fleet_manager.get_node(node_id)
            if node is None:
                # Invalid node ID, job waits
                continue

            # Check if node can accept job
            if not node.can_accept_job(job.required_memory):
                # Node not suitable, job waits
                continue

            # SIMULATE CLOUD PROVISIONING DELAY
            # Check if fleet manager supports on-demand provisioning
            provisioning_delay = 0.0
            if hasattr(self.fleet_manager, 'request_provisioning'):
                provisioning_delay = self.fleet_manager.request_provisioning(node_id, self.current_time)
            
            # Actual start time is after provisioning delay
            actual_start_time = self.current_time + provisioning_delay

            # Assign job to node
            success = self.fleet_manager.assign_job_to_node(
                node_id=node_id,
                job_id=job.job_id,
                current_time=actual_start_time,
            )

            if success:
                self.job_queue.remove(job.job_id)
                # Update job state
                job.state = JobState.RUNNING
                job.start_time = actual_start_time
                job.assigned_node = node_id

                # Get execution time for this job on this node
                from cgrsp.data.execution_times import ExecutionTimeModel, GPUType

                gpu_type_map = {
                    "RTX_3090": GPUType.RTX_3090,
                    "RTX_A5000": GPUType.RTX_A5000,
                    "RTX_A4500": GPUType.RTX_A4500,
                    "RTX_A4000": GPUType.RTX_A4000,
                }

                gpu_type = gpu_type_map.get(node.resource_type.value, GPUType.RTX_3090)
                complexity = int(job.complexity if job.true_complexity is None else job.true_complexity)
                # Realised execution time is a log-normal draw with expected value
                # equal to the empirical stratum mean W(kappa, g). Schedulers still
                # plan with the deterministic mean (the estimate e_j); only the
                # realised completion time is stochastic, matching real operation.
                execution_time = ExecutionTimeModel.sample_execution_time(
                    complexity, gpu_type, self._exec_rng
                )

                # Record metrics (start_time includes provisioning delay)
                self.metrics_collector.record_job_start(
                    job_id=job.job_id,
                    start_time=actual_start_time,
                    node_id=node_id,
                    resource_type=node.resource_type.value,
                    dispatch_time=self.current_time,
                )

                # Schedule completion event
                completion_time = actual_start_time + execution_time
                self.schedule_event(
                    EventType.JOB_COMPLETION,
                    completion_time,
                    {
                        "node_id": node_id,
                        "job_id": job.job_id,
                    },
                )

    def _default_scheduler(self) -> None:
        """Apply FIFO ordering with the same dispatch model as explicit policies."""
        from cgrsp.schedulers.fifo_scheduler import FIFOScheduler

        jobs = self.job_queue.get_jobs_sorted_by_priority()
        assignments = FIFOScheduler(self.fleet_manager).schedule(jobs, self.current_time)
        self._dispatch_assignments(jobs, assignments)

    def set_scheduler(self, scheduler) -> None:
        """
        Register a custom scheduler.

        Args:
            scheduler: Scheduler object with schedule() method or function
        """
        self.scheduler_object = scheduler
        # Keep backward compatibility with function-based schedulers
        if callable(scheduler) and not hasattr(scheduler, 'schedule'):
            self.scheduler = scheduler
        else:
            self.scheduler = None  # Will use scheduler_object in _custom_scheduler

    def run(self, duration: float, warmup_period: float = 0.0,
            drain: bool = False) -> None:
        """
        Run the simulation for a specified duration.

        Args:
            duration: Simulation duration in seconds
            warmup_period: Exclude jobs submitted before this offset into the run
                from aggregate metrics, while retaining their effect on the fleet
            drain: Finish queued and running work after the observation window
        """
        if duration < 0 or not 0 <= warmup_period <= duration:
            raise ValueError("Require 0 <= warmup_period <= duration")
        self.metrics_collector.set_observation_start(self.current_time + warmup_period)
        draining = False
        # Schedule simulation end
        self.schedule_event(
            EventType.SIMULATION_END,
            self.current_time + duration,
            {},
            priority=100,  # High priority
        )

        # Schedule initial node transitions
        for node in self.fleet_manager.get_all_nodes():
            time_to_next = self.fleet_manager.calculate_time_to_next_transition(
                node_id=node.node_id,
                current_time=self.current_time,
            )

            if time_to_next < float('inf'):
                self.schedule_event(
                    EventType.NODE_TRANSITION,
                    self.current_time + time_to_next,
                    {"node_id": node.node_id},
                )

        self.running = True

        # Main event loop
        while self.running and self._event_queue:
            event = self._pop_next_event()

            if event is None:
                break

            # Advance time to event
            self.current_time = event.timestamp

            # Record time-series data
            self.metrics_collector.record_time_series_state(
                current_time=self.current_time,
                queue_length=self.job_queue.get_queue_length(),
                idle_nodes=self.fleet_manager.get_fleet_size() - len(self.fleet_manager.get_nodes_by_state(NodeState.RUNNING)) - len(self.fleet_manager.get_nodes_by_state(NodeState.UNAVAILABLE)),
                running_nodes=len(self.fleet_manager.get_nodes_by_state(NodeState.RUNNING)),
                unavailable_nodes=len(self.fleet_manager.get_nodes_by_state(NodeState.UNAVAILABLE)),
            )

            # Record utilization
            self.metrics_collector.record_utilization_sample(
                current_time=self.current_time,
                utilization=self.fleet_manager.get_fleet_utilization(self.current_time),
            )

            # Process event
            if event.event_type == EventType.JOB_ARRIVAL:
                if draining:
                    continue
                self._handle_job_arrival(event)

            elif event.event_type == EventType.JOB_COMPLETION:
                self._handle_job_completion(event)

            elif event.event_type == EventType.NODE_TRANSITION:
                self._handle_node_transition(event)

            elif event.event_type == EventType.SCHEDULER_DECISION:
                self._handle_scheduler_decision(event)

            elif event.event_type == EventType.SIMULATION_END:
                draining = drain
                if not draining:
                    self.running = False
                    break

            if draining and self.job_queue.is_empty() and not self.fleet_manager.get_nodes_by_state(NodeState.RUNNING):
                self.running = False
                break

        if drain and (not self.job_queue.is_empty() or self.fleet_manager.get_nodes_by_state(NodeState.RUNNING)):
            raise RuntimeError("Simulation ended before the backlog drained")

    def get_metrics_summary(self) -> Dict:
        """
        Get a summary of collected metrics.

        Returns:
            Dictionary with key metrics
        """
        return self.metrics_collector.generate_summary_report()

    def reset(self, keep_config: bool = True) -> None:
        """
        Reset the simulator state.

        Args:
            keep_config: If True, preserve fleet and temporal configuration
        """
        if not keep_config:
            raise NotImplementedError("Full reset not implemented. Create a new Simulator instance.")

        self.current_time = 0.0
        self._next_job_id = 0
        self._event_queue.clear()
        self._event_counter = 0
        self.running = False

        self.job_queue.clear()
        self.metrics_collector.reset()

        # Reset all nodes to idle
        for node in self.fleet_manager.get_all_nodes():
            node.state = NodeState.IDLE
            node.current_job_id = None
            node.unavailable_reason = None
            node.state_entry_time = 0.0

        self._update_observable_state()
