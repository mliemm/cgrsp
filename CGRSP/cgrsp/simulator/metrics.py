"""
Metrics Collector Module

Tracks and aggregates QoS metrics, operational costs, and performance
indicators for the CGRSP simulator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import statistics


@dataclass
class JobMetrics:
    """
    Metrics collected for a single job.

    Attributes:
        job_id: Job identifier
        submit_time: Job submission time
        start_time: Job start execution time
        completion_time: Job completion time
        deadline: Job deadline
        waiting_time: Time spent waiting in queue (w_j)
        response_time: Total time from submission to completion (r_j)
        execution_time: Actual execution time
        deadline_met: Whether job met its deadline (δ_j)
        execution_cost: Cost of executing the job (c_j)
        assigned_node_id: Node that executed the job
        resource_type: GPU resource type used
    """

    job_id: int
    submit_time: float
    deadline: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    waiting_time: float = 0.0
    response_time: float = 0.0
    execution_time: float = 0.0
    deadline_met: bool = True
    execution_cost: float = 0.0
    assigned_node_id: Optional[int] = None
    resource_type: Optional[str] = None

    def calculate_metrics(self) -> None:
        """Calculate all metrics for a completed job."""
        if self.start_time is not None and self.completion_time is not None:
            self.waiting_time = max(0.0, self.start_time - self.submit_time)
            self.execution_time = self.completion_time - self.start_time
            self.response_time = self.completion_time - self.submit_time
            self.deadline_met = self.completion_time <= self.deadline

    def get_tardiness(self) -> float:
        """
        Calculate job tardiness (positive if late, zero otherwise).

        Returns:
            Tardiness in seconds
        """
        if self.completion_time is None:
            return 0.0

        return max(0.0, self.completion_time - self.deadline)

    def get_lateness(self) -> float:
        """
        Calculate job lateness (can be negative if early).

        Returns:
            Lateness in seconds (completion_time - deadline)
        """
        if self.completion_time is None:
            return 0.0

        return self.completion_time - self.deadline


@dataclass
class SystemMetrics:
    """
    Aggregate metrics for the entire system over a time period.
    """

    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_deadline_missed: int = 0
    total_execution_cost: float = 0.0

    # Timing statistics
    total_waiting_time: float = 0.0
    total_response_time: float = 0.0
    total_execution_time: float = 0.0
    total_tardiness: float = 0.0

    # Per-job lists for detailed statistics
    waiting_times: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)
    tardiness_values: List[float] = field(default_factory=list)
    job_costs: List[float] = field(default_factory=list)
    objective_values: List[float] = field(default_factory=list)  # per-job combined objective

    def get_deadline_miss_rate(self) -> float:
        """Calculate deadline miss rate."""
        if self.total_jobs_completed == 0:
            return 0.0
        return self.total_jobs_deadline_missed / self.total_jobs_completed

    def get_average_waiting_time(self) -> float:
        """Calculate average waiting time."""
        if not self.waiting_times:
            return 0.0
        return statistics.mean(self.waiting_times)

    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def get_average_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.execution_times:
            return 0.0
        return statistics.mean(self.execution_times)

    def get_average_tardiness(self) -> float:
        """Calculate average tardiness."""
        if not self.tardiness_values:
            return 0.0
        return statistics.mean(self.tardiness_values)

    def get_median_waiting_time(self) -> float:
        """Calculate median waiting time."""
        if not self.waiting_times:
            return 0.0
        return statistics.median(self.waiting_times)

    def get_median_response_time(self) -> float:
        """Calculate median response time."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    def get_waiting_time_percentile(self, percentile: float) -> float:
        """
        Calculate waiting time percentile.

        Args:
            percentile: Percentile value (0-100)

        Returns:
            Waiting time at percentile
        """
        if not self.waiting_times:
            return 0.0
        return statistics.quantiles(self.waiting_times, n=100)[int(percentile) - 1]

    def get_response_time_percentile(self, percentile: float) -> float:
        """
        Calculate response time percentile.

        Args:
            percentile: Percentile value (0-100)

        Returns:
            Response time at percentile
        """
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=100)[int(percentile) - 1]

    def get_average_cost_per_job(self) -> float:
        """Calculate average cost per job."""
        if self.total_jobs_completed == 0:
            return 0.0
        return self.total_execution_cost / self.total_jobs_completed

    def get_completion_rate(self) -> float:
        """Calculate job completion rate."""
        if self.total_jobs_submitted == 0:
            return 0.0
        return self.total_jobs_completed / self.total_jobs_submitted


class MetricsCollector:
    """
    Collects and aggregates metrics from simulation runs.

    Tracks per-job metrics and system-wide aggregates in real-time.
    """

    def __init__(self):
        """Initialize an empty metrics collector."""
        self.observation_start = 0.0
        # Per-job metrics storage
        self._job_metrics: Dict[int, JobMetrics] = {}

        # System-wide aggregates
        self.system_metrics = SystemMetrics()

        # Time-series tracking
        self._time_series: List[tuple[float, Dict]] = []

        # Resource utilization tracking
        self._utilization_samples: List[tuple[float, float]] = []

        # Dispatch log: one (job_id, node_id, dispatch_time, start_time) entry per
        # record_job_start call, never overwritten, so the constraint verifier can
        # count dispatches per job and rebuild each slot's occupancy intervals.
        self.dispatch_log: List[tuple] = []

    def record_job_submission(self, job_id: int, submit_time: float, deadline: float) -> None:
        """
        Record a job submission.

        Args:
            job_id: Job identifier
            submit_time: Submission time
            deadline: Job deadline
        """
        self._job_metrics[job_id] = JobMetrics(
            job_id=job_id,
            submit_time=submit_time,
            deadline=deadline,
        )
        if submit_time >= self.observation_start:
            self.system_metrics.total_jobs_submitted += 1

    def set_observation_start(self, timestamp: float) -> None:
        """Select the submission cohort and rebuild aggregates for existing jobs."""
        self.observation_start = timestamp
        self.system_metrics = SystemMetrics()
        for job in self._job_metrics.values():
            if job.submit_time >= timestamp:
                self.system_metrics.total_jobs_submitted += 1
                if job.completion_time is not None:
                    self.record_job_completion(job.job_id, job.completion_time, job.execution_cost)

    def record_job_start(
        self,
        job_id: int,
        start_time: float,
        node_id: int,
        resource_type: str,
        dispatch_time: Optional[float] = None,
    ) -> None:
        """
        Record a job starting execution.

        Args:
            job_id: Job identifier
            start_time: Start time
            node_id: Assigned node ID
            resource_type: GPU resource type
            dispatch_time: Slot-acquisition time (before any provisioning delay);
                defaults to start_time when there is no provisioning delay
        """
        self.dispatch_log.append(
            (job_id, node_id, start_time if dispatch_time is None else dispatch_time, start_time)
        )
        if job_id not in self._job_metrics:
            return

        job = self._job_metrics[job_id]
        job.start_time = start_time
        job.assigned_node_id = node_id
        job.resource_type = resource_type

    def record_job_completion(
        self,
        job_id: int,
        completion_time: float,
        execution_cost: float
    ) -> None:
        """
        Record a job completion.

        Args:
            job_id: Job identifier
            completion_time: Completion time
            execution_cost: Total execution cost
        """
        if job_id not in self._job_metrics:
            return

        job = self._job_metrics[job_id]
        job.completion_time = completion_time
        job.execution_cost = execution_cost
        job.calculate_metrics()

        if job.submit_time < self.observation_start:
            return

        # Update system metrics
        self.system_metrics.total_jobs_completed += 1
        self.system_metrics.total_execution_cost += execution_cost
        self.system_metrics.total_waiting_time += job.waiting_time
        self.system_metrics.total_response_time += job.response_time
        self.system_metrics.total_execution_time += job.execution_time
        self.system_metrics.total_tardiness += job.get_tardiness()

        if not job.deadline_met:
            self.system_metrics.total_jobs_deadline_missed += 1

        # Add to lists for statistical analysis
        self.system_metrics.waiting_times.append(job.waiting_time)
        self.system_metrics.response_times.append(job.response_time)
        self.system_metrics.execution_times.append(job.execution_time)
        self.system_metrics.tardiness_values.append(job.get_tardiness())
        self.system_metrics.job_costs.append(execution_cost)

        # Combined objective: symmetric weighted-sum robustness variant of the
        # lexicographic QoS-primary objective (Eq. objective in the formulation).
        # The QoS term uses the binary deadline-miss indicator (1 - delta_j),
        # matching the model: deadline violations are penalised by whether a job
        # misses, not by continuous tardiness (tardiness is a reported diagnostic
        # only, never part of the objective).
        W_QOS, ALPHA1, ALPHA2, W_COST = 0.5, 1.0, 10.0, 0.5
        miss_indicator = 0.0 if job.deadline_met else 1.0
        obj = W_QOS * (ALPHA1 * job.waiting_time + ALPHA2 * miss_indicator) + W_COST * execution_cost
        self.system_metrics.objective_values.append(obj)

    def record_utilization_sample(self, current_time: float, utilization: float) -> None:
        """
        Record a system utilization sample.

        Args:
            current_time: Current simulation time
            utilization: System utilization (0-1)
        """
        self._utilization_samples.append((current_time, utilization))

    def record_time_series_state(
        self,
        current_time: float,
        queue_length: int,
        idle_nodes: int,
        running_nodes: int,
        unavailable_nodes: int
    ) -> None:
        """
        Record system state for time-series analysis.

        Args:
            current_time: Current simulation time
            queue_length: Number of jobs in queue
            idle_nodes: Number of idle nodes
            running_nodes: Number of running nodes
            unavailable_nodes: Number of unavailable nodes
        """
        self._time_series.append({
            "time": current_time,
            "queue_length": queue_length,
            "idle_nodes": idle_nodes,
            "running_nodes": running_nodes,
            "unavailable_nodes": unavailable_nodes,
        })

    def get_job_metrics(self, job_id: int) -> Optional[JobMetrics]:
        """Get metrics for a specific job."""
        return self._job_metrics.get(job_id)

    def get_all_job_metrics(self) -> List[JobMetrics]:
        """Get metrics for all jobs."""
        return list(self._job_metrics.values())

    def get_completed_job_metrics(self, include_warmup: bool = False) -> List[JobMetrics]:
        """Get completed records, optionally including warmup for physical checks."""
        return [
            job for job in self._job_metrics.values()
            if job.completion_time is not None
            and (include_warmup or job.submit_time >= self.observation_start)
        ]

    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide aggregated metrics."""
        return self.system_metrics

    def get_average_utilization(self) -> float:
        """Calculate average system utilization over time."""
        samples = self.get_utilization_time_series()
        if not samples:
            return 0.0
        return statistics.mean(sample[1] for sample in samples)

    def get_time_series(self) -> List[Dict]:
        """Get time-series data."""
        return [sample for sample in self._time_series if sample["time"] >= self.observation_start]

    def get_utilization_time_series(self) -> List[tuple[float, float]]:
        """Get utilization time-series data."""
        return [sample for sample in self._utilization_samples if sample[0] >= self.observation_start]

    @staticmethod
    def _time_stats(values: List[float]) -> Dict:
        """Return a full stats block for a list of durations (all in sec/min/hr)."""
        if not values:
            zero = {"sec": 0.0, "min": 0.0, "hr": 0.0}
            return {
                "avg": zero.copy(), "median": zero.copy(),
                "std_dev_sec": 0.0,
                "min": zero.copy(), "max": zero.copy(),
                "p95": {"sec": None, "min": None},
                "p99": {"sec": None, "min": None},
                "total": zero.copy(),
                "count": 0,
            }

        avg = statistics.mean(values)
        med = statistics.median(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        mn = min(values)
        mx = max(values)
        total = sum(values)
        n = len(values)

        def tri(s):
            return {"sec": round(s, 4), "min": round(s / 60, 4), "hr": round(s / 3600, 6)}

        def p_or_none(pct):
            if n < pct:
                return {"sec": None, "min": None}
            v = statistics.quantiles(values, n=100)[int(pct) - 1]
            return {"sec": round(v, 4), "min": round(v / 60, 4)}

        return {
            "avg": tri(avg),
            "median": tri(med),
            "std_dev_sec": round(std, 4),
            "min": tri(mn),
            "max": tri(mx),
            "p95": p_or_none(95),
            "p99": p_or_none(99),
            "total": tri(total),
            "count": n,
        }

    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report with all metrics in sec/min/hr.

        Returns:
            Dictionary with all key metrics and explicit unit suffixes
        """
        sm = self.system_metrics
        util_values = [u for _, u in self.get_utilization_time_series()]

        cost_stats: Dict = {}
        if sm.job_costs:
            cost_stats = {
                "total_usd": round(sm.total_execution_cost, 6),
                "avg_per_job_usd": round(statistics.mean(sm.job_costs), 6),
                "median_per_job_usd": round(statistics.median(sm.job_costs), 6),
                "std_dev_per_job_usd": round(statistics.stdev(sm.job_costs) if len(sm.job_costs) > 1 else 0.0, 6),
                "min_per_job_usd": round(min(sm.job_costs), 6),
                "max_per_job_usd": round(max(sm.job_costs), 6),
            }
        else:
            cost_stats = {
                "total_usd": 0.0,
                "avg_per_job_usd": 0.0,
                "median_per_job_usd": 0.0,
                "std_dev_per_job_usd": 0.0,
                "min_per_job_usd": 0.0,
                "max_per_job_usd": 0.0,
            }

        util_stats: Dict = {}
        if util_values:
            util_stats = {
                "avg_pct": round(statistics.mean(util_values) * 100, 4),
                "median_pct": round(statistics.median(util_values) * 100, 4),
                "min_pct": round(min(util_values) * 100, 4),
                "max_pct": round(max(util_values) * 100, 4),
                "std_dev_pct": round((statistics.stdev(util_values) if len(util_values) > 1 else 0.0) * 100, 4),
            }
        else:
            util_stats = {"avg_pct": 0.0, "median_pct": 0.0, "min_pct": 0.0, "max_pct": 0.0, "std_dev_pct": 0.0}

        completion_rate = sm.get_completion_rate()
        deadline_miss_rate = sm.get_deadline_miss_rate()

        # Combined objective function value
        obj_total = sum(sm.objective_values)
        obj_avg = (obj_total / len(sm.objective_values)) if sm.objective_values else 0.0

        # GPU type breakdown: count assignments per resource type
        from collections import Counter
        gpu_counts = Counter(
            job.resource_type
            for job in self._job_metrics.values()
            if job.resource_type is not None and job.submit_time >= self.observation_start
        )

        tard = sm.tardiness_values
        avg_tardiness_s = statistics.mean(tard) if tard else 0.0
        p95_tardiness_s = (
            sorted(tard)[max(0, int(round(0.95 * len(tard))) - 1)]
            if tard else 0.0
        )

        return {
            "job_counts": {
                "submitted": sm.total_jobs_submitted,
                "completed": sm.total_jobs_completed,
                "pending": sm.total_jobs_submitted - sm.total_jobs_completed,
                "deadline_missed": sm.total_jobs_deadline_missed,
                "deadline_met": sm.total_jobs_completed - sm.total_jobs_deadline_missed,
                "completion_rate_pct": round(completion_rate * 100, 4),
                "deadline_miss_rate_pct": round(deadline_miss_rate * 100, 4),
                "deadline_met_rate_pct": round((1.0 - deadline_miss_rate) * 100, 4),
            },
            "avg_tardiness_s": avg_tardiness_s,
            "p95_tardiness_s": p95_tardiness_s,
            "waiting_time": self._time_stats(sm.waiting_times),
            "response_time": self._time_stats(sm.response_times),
            "execution_time": self._time_stats(sm.execution_times),
            "tardiness": self._time_stats(sm.tardiness_values),
            "cost": cost_stats,
            "utilization": util_stats,
            "objective": {
                "total": round(obj_total, 4),
                "avg_per_job": round(obj_avg, 6),
                "description": "0.5*(1.0*wait_s + 10.0*miss_indicator) + 0.5*cost_usd",
            },
            "gpu_breakdown": dict(gpu_counts),
            # Legacy keys kept for backward compat with callers that use old keys
            "timing_metrics": {
                "avg_waiting_time": sm.get_average_waiting_time(),
                "median_waiting_time": sm.get_median_waiting_time(),
                "avg_response_time": sm.get_average_response_time(),
                "median_response_time": sm.get_median_response_time(),
                "avg_execution_time": sm.get_average_execution_time(),
                "avg_tardiness": sm.get_average_tardiness(),
            },
            "cost_metrics": {
                "total_cost": sm.total_execution_cost,
                "avg_cost_per_job": sm.get_average_cost_per_job(),
            },
            "utilization_legacy": {
                "average": self.get_average_utilization(),
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.observation_start = 0.0
        self._job_metrics.clear()
        self.system_metrics = SystemMetrics()
        self._time_series.clear()
        self._utilization_samples.clear()
        self.dispatch_log.clear()

    def export_to_csv(self, filepath: str) -> None:
        """
        Export job metrics to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'job_id', 'submit_time', 'start_time', 'completion_time',
                'deadline', 'waiting_time', 'response_time', 'execution_time',
                'deadline_met', 'execution_cost', 'assigned_node_id', 'resource_type',
                'tardiness', 'lateness'
            ])

            for job in self.get_all_job_metrics():
                writer.writerow([
                    job.job_id,
                    job.submit_time,
                    job.start_time if job.start_time is not None and job.start_time >= 0 else '',
                    job.completion_time if job.completion_time is not None and job.completion_time >= 0 else '',
                    job.deadline,
                    f"{job.waiting_time:.2f}",
                    f"{job.response_time:.2f}",
                    f"{job.execution_time:.2f}",
                    job.deadline_met,
                    f"{job.execution_cost:.2f}",
                    job.assigned_node_id if job.assigned_node_id is not None else '',
                    job.resource_type if job.resource_type else '',
                    f"{job.get_tardiness():.2f}",
                    f"{job.get_lateness():.2f}",
                ])
