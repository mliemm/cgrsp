"""
Verification Tests Module

Verifies that scheduling algorithms respect all constraints
including deadlines, assignments, and cost calculations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from cgrsp.simulator.core import Simulator
from cgrsp.simulator.queue import Job, JobState
from cgrsp.simulator.fleet import Node, NodeState
from cgrsp.config import ResourceType, RESOURCE_SPECS


class VerificationStatus(str, Enum):
    """Status of verification checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "n/a"


@dataclass
class ConstraintViolation:
    """
    Represents a constraint violation.

    Attributes:
        constraint_name: Name of violated constraint
        violation_type: Type of violation
        description: Human-readable description
        severity: Severity level (error, warning)
        affected_jobs: List of affected job IDs
        affected_nodes: List of affected node IDs
    """
    constraint_name: str
    violation_type: str
    description: str
    severity: str = "error"
    affected_jobs: List[int] = field(default_factory=list)
    affected_nodes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "constraint_name": self.constraint_name,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "affected_jobs": self.affected_jobs,
            "affected_nodes": self.affected_nodes,
        }


@dataclass
class VerificationReport:
    """
    Report from verification testing.

    Attributes:
        all_passed: Whether all checks passed
        total_violations: Total number of violations found
        violations: List of all violations
        constraint_results: Results for each constraint type
    """
    all_passed: bool = True
    total_violations: int = 0
    violations: List[ConstraintViolation] = field(default_factory=list)
    constraint_results: Dict[str, Dict] = field(default_factory=dict)

    def add_violation(self, violation: ConstraintViolation) -> None:
        """Add a violation to the report."""
        self.violations.append(violation)
        self.total_violations += 1
        if violation.severity == "error":
            self.all_passed = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "all_passed": self.all_passed,
            "total_violations": self.total_violations,
            "violations": [v.to_dict() for v in self.violations],
            "constraint_results": self.constraint_results,
        }

    def print_summary(self) -> None:
        """Print verification summary to console."""
        print("\n" + "="*80)
        print("VERIFICATION REPORT")
        print("="*80)

        status = "PASSED" if self.all_passed else "FAILED"
        print(f"\nOverall Status: {status}")
        print(f"Total Violations: {self.total_violations}")

        if self.violations:
            print("\nViolations Found:")
            for i, violation in enumerate(self.violations, 1):
                print(f"\n{i}. {violation.constraint_name}")
                print(f"   Type: {violation.violation_type}")
                print(f"   Severity: {violation.severity}")
                print(f"   Description: {violation.description}")
                if violation.affected_jobs:
                    print(f"   Affected Jobs: {violation.affected_jobs[:10]}")
                    if len(violation.affected_jobs) > 10:
                        print(f"      ... and {len(violation.affected_jobs) - 10} more")
                if violation.affected_nodes:
                    print(f"   Affected Nodes: {violation.affected_nodes[:10]}")
                    if len(violation.affected_nodes) > 10:
                        print(f"      ... and {len(violation.affected_nodes) - 10} more")

        print("\n" + "="*80)


class DeadlineVerifier:
    """
    Verify deadline enforcement constraints.

    Constraints:
    - Each job's recorded deadline_met flag agrees with its own recorded
      completion time and its own deadline d_j
    - The reported fleet-wide deadline-miss count agrees with a fresh recount
      over the per-job records

    Deadlines are per job: d_j is absolute and heterogeneous (the main
    experiments mix a 1 h tight window with an 8 h loose one), so no single
    fleet-wide threshold applies and none is accepted here.
    """

    def verify(self, simulator: Simulator) -> VerificationReport:
        """
        Verify deadline constraints.

        Args:
            simulator: Simulator instance to verify

        Returns:
            VerificationReport with deadline violations
        """
        report = VerificationReport()
        report.constraint_results["deadline_enforcement"] = {
            "checked": True,
            "description": "Verify each job's deadline_met flag and the reported miss count",
        }

        # Read deadlines from retained per-job metrics. Queue entries are
        # removed at dispatch and can be discarded during compaction, so the
        # queue is not a history of completed jobs.
        completed_jobs = simulator.metrics_collector.get_completed_job_metrics()

        deadline_missed_jobs = []
        deadline_met_but_late = []
        recounted_misses = 0

        for job_metrics in completed_jobs:
            deadline = job_metrics.deadline
            completion_time = job_metrics.completion_time

            if deadline is None or completion_time is None:
                continue

            if completion_time > deadline:
                recounted_misses += 1
                if job_metrics.deadline_met:
                    # Violation: missed deadline but not marked
                    deadline_missed_jobs.append(job_metrics.job_id)
            elif not job_metrics.deadline_met:
                # Warning: marked as missed but completed on time
                deadline_met_but_late.append(job_metrics.job_id)

        if deadline_missed_jobs:
            report.add_violation(ConstraintViolation(
                constraint_name="deadline_enforcement",
                violation_type="missed_deadline_not_marked",
                description=f"{len(deadline_missed_jobs)} jobs missed deadline but were not marked as deadline_missed",
                severity="error",
                affected_jobs=deadline_missed_jobs,
            ))

        if deadline_met_but_late:
            report.add_violation(ConstraintViolation(
                constraint_name="deadline_enforcement",
                violation_type="incorrect_deadline_flag",
                description=f"{len(deadline_met_but_late)} jobs marked as missed but completed on time",
                severity="warning",
                affected_jobs=deadline_met_but_late,
            ))

        # The reported miss rate is a headline result, and it is accumulated on
        # a separate path from the per-job records, so recount and compare.
        reported_misses = simulator.metrics_collector.system_metrics.total_jobs_deadline_missed
        if reported_misses != recounted_misses:
            report.add_violation(ConstraintViolation(
                constraint_name="deadline_enforcement",
                violation_type="miss_count_mismatch",
                description=(
                    f"Reported deadline-miss count ({reported_misses}) disagrees with a "
                    f"recount over the per-job records ({recounted_misses})"
                ),
                severity="error",
            ))

        report.constraint_results["deadline_enforcement"]["status"] = (
            "passed" if report.all_passed else "failed"
        )

        return report


class AssignmentVerifier:
    """
    Verify assignment uniqueness constraints.

    Constraints:
    - Each completed job dispatched exactly once
    - Each node running at most one job
    """

    def verify(self, simulator: Simulator) -> VerificationReport:
        """
        Verify assignment constraints.

        Args:
            simulator: Simulator instance to verify

        Returns:
            VerificationReport with assignment violations
        """
        report = VerificationReport()
        report.constraint_results["assignment_uniqueness"] = {
            "checked": True,
            "description": "Verify each job dispatched once and each node holding ≤1 job at a time (dispatch log, [dispatch, completion) occupancy)",
        }

        # Dispatch history (Constraints eq:single_assign and eq:single_job_node): the
        # collector's dispatch log keeps every record_job_start call, so each completed
        # job must have been dispatched exactly once, no job may be dispatched twice,
        # and no slot may hold two jobs whose occupancy intervals
        # [dispatch, completion) overlap (provisioning wait included). A job still
        # held at the end of the run occupies its slot from dispatch onwards.
        mc = simulator.metrics_collector
        completed = {jm.job_id: jm for jm in mc.get_completed_job_metrics(include_warmup=True)}
        observed_completed = mc.get_completed_job_metrics()
        dispatches: Dict[int, List] = {}
        for job_id, node_id, t_dispatch, _t_start in mc.dispatch_log:
            dispatches.setdefault(job_id, []).append((node_id, t_dispatch))
        history: Dict[int, List] = {}
        for job_id in sorted(set(completed) | set(dispatches)):
            ds = dispatches.get(job_id, [])
            if len(ds) != 1:
                report.add_violation(ConstraintViolation(
                    constraint_name="assignment_uniqueness",
                    violation_type="job_multiple_nodes",
                    description=f"Job {job_id} dispatched {len(ds)} times (nodes {[n for n, _ in ds]})",
                    severity="error",
                    affected_jobs=[job_id],
                    affected_nodes=sorted({n for n, _ in ds}),
                ))
            end = completed[job_id].completion_time if job_id in completed else float("inf")
            for node_id, t_dispatch in ds:
                history.setdefault(node_id, []).append((t_dispatch, end, job_id))
        if mc.system_metrics.total_jobs_completed != len(observed_completed):
            report.add_violation(ConstraintViolation(
                constraint_name="assignment_uniqueness",
                violation_type="completion_count_mismatch",
                description=f"{mc.system_metrics.total_jobs_completed} completions counted for {len(observed_completed)} observed completed jobs",
                severity="error",
                affected_jobs=[],
                affected_nodes=[],
            ))
        for node_id, runs in history.items():
            runs.sort()
            for (s0, e0, j0), (s1, e1, j1) in zip(runs, runs[1:]):
                if s1 < e0 - 1e-9:
                    report.add_violation(ConstraintViolation(
                        constraint_name="assignment_uniqueness",
                        violation_type="node_multiple_jobs",
                        description=f"Node {node_id} held jobs {j0} and {j1} with overlapping occupancy intervals",
                        severity="error",
                        affected_jobs=[j0, j1],
                        affected_nodes=[node_id],
                    ))
        report.constraint_results["assignment_uniqueness"]["jobs_checked"] = len(completed)

        # Check job assignments
        job_assignments: Dict[int, List[int]] = {}  # job_id -> list of node_ids
        node_assignments: Dict[int, List[int]] = {}  # node_id -> list of job_ids

        # Check all nodes
        for node in simulator.fleet_manager.get_all_nodes():
            if node.state == NodeState.RUNNING and node.current_job_id is not None:
                if node.current_job_id not in job_assignments:
                    job_assignments[node.current_job_id] = []
                job_assignments[node.current_job_id].append(node.node_id)

                if node.node_id not in node_assignments:
                    node_assignments[node.node_id] = []
                node_assignments[node.node_id].append(node.current_job_id)

                # Check if node has multiple jobs
                if len(node_assignments[node.node_id]) > 1:
                    report.add_violation(ConstraintViolation(
                        constraint_name="assignment_uniqueness",
                        violation_type="node_multiple_jobs",
                        description=f"Node {node.node_id} assigned to multiple jobs: {node_assignments[node.node_id]}",
                        severity="error",
                        affected_jobs=node_assignments[node.node_id],
                        affected_nodes=[node.node_id],
                    ))

        # Check if jobs assigned to multiple nodes
        for job_id, node_ids in job_assignments.items():
            if len(node_ids) > 1:
                report.add_violation(ConstraintViolation(
                    constraint_name="assignment_uniqueness",
                    violation_type="job_multiple_nodes",
                    description=f"Job {job_id} assigned to multiple nodes: {node_ids}",
                    severity="error",
                    affected_jobs=[job_id],
                    affected_nodes=node_ids,
                ))

        report.constraint_results["assignment_uniqueness"]["status"] = (
            "passed" if report.all_passed else "failed"
        )

        return report


class CostVerifier:
    """
    Verify cost calculation constraints.

    Constraints:
    - Job cost = execution_time × hourly_rate × time_scaling
    - Hourly rate matches resource specification
    """

    def verify(self, simulator: Simulator) -> VerificationReport:
        """
        Verify cost calculation constraints.

        Args:
            simulator: Simulator instance to verify

        Returns:
            VerificationReport with cost violations
        """
        report = VerificationReport()
        report.constraint_results["cost_calculation"] = {
            "checked": True,
            "description": "Verify cost = W(j,g) × p_g (time-weighted)",
        }

        completed_jobs = simulator.metrics_collector.get_completed_job_metrics()
        cost_errors = []

        for job_metrics in completed_jobs:
            if job_metrics.assigned_node_id is None or job_metrics.resource_type is None:
                continue

            # Get node to calculate expected cost
            node = simulator.fleet_manager.get_node(job_metrics.assigned_node_id)
            if node is None:
                continue

            # Calculate expected cost
            if job_metrics.start_time is not None and job_metrics.completion_time is not None:
                execution_time_hours = (job_metrics.completion_time - job_metrics.start_time) / 3600.0
                hourly_rate = node.get_hourly_cost()
                expected_cost = execution_time_hours * hourly_rate

                # Compare with recorded cost (allow small floating point error)
                actual_cost = job_metrics.execution_cost
                cost_diff = abs(expected_cost - actual_cost)

                # Allow 1% tolerance for floating point errors
                if expected_cost > 0 and cost_diff / expected_cost > 0.01:
                    cost_errors.append({
                        "job_id": job_metrics.job_id,
                        "expected_cost": expected_cost,
                        "actual_cost": actual_cost,
                        "difference": cost_diff,
                    })

        if cost_errors:
            affected_job_ids = [e["job_id"] for e in cost_errors]
            report.add_violation(ConstraintViolation(
                constraint_name="cost_calculation",
                violation_type="cost_mismatch",
                description=f"{len(cost_errors)} jobs have incorrect cost calculations",
                severity="error",
                affected_jobs=affected_job_ids,
            ))

        report.constraint_results["cost_calculation"]["status"] = (
            "passed" if report.all_passed else "failed"
        )

        return report


class ConstraintVerifier:
    """
    Main verifier that runs all constraint checks.
    """

    def __init__(self):
        """Initialize constraint verifier."""
        self.deadline_verifier = DeadlineVerifier()
        self.assignment_verifier = AssignmentVerifier()
        self.cost_verifier = CostVerifier()

    def verify_all(self, simulator: Simulator) -> VerificationReport:
        """
        Run all verification checks.

        Args:
            simulator: Simulator instance to verify

        Returns:
            Combined VerificationReport
        """
        combined_report = VerificationReport()

        # Run all verifiers
        deadline_report = self.deadline_verifier.verify(simulator)
        assignment_report = self.assignment_verifier.verify(simulator)
        cost_report = self.cost_verifier.verify(simulator)

        # Combine reports
        combined_report.violations.extend(deadline_report.violations)
        combined_report.violations.extend(assignment_report.violations)
        combined_report.violations.extend(cost_report.violations)

        combined_report.total_violations = (
            deadline_report.total_violations +
            assignment_report.total_violations +
            cost_report.total_violations
        )

        combined_report.all_passed = (
            deadline_report.all_passed and
            assignment_report.all_passed and
            cost_report.all_passed
        )

        combined_report.constraint_results = {
            **deadline_report.constraint_results,
            **assignment_report.constraint_results,
            **cost_report.constraint_results,
        }

        return combined_report

    def verify_job_states(self, simulator: Simulator) -> VerificationReport:
        """
        Verify job state consistency.

        Checks:
        - All running jobs have start_time and assigned_node
        - All completed jobs have completion_time
        - Job states match actual status

        Args:
            simulator: Simulator instance to verify

        Returns:
            VerificationReport with state violations
        """
        report = VerificationReport()
        report.constraint_results["job_state_consistency"] = {
            "checked": True,
            "description": "Verify job states match actual status",
        }

        all_jobs = simulator.job_queue.get_all_jobs()
        state_errors = []

        for job in all_jobs:
            # Check running jobs
            if job.state == JobState.RUNNING:
                if job.start_time is None:
                    state_errors.append({
                        "job_id": job.job_id,
                        "issue": "running but no start_time",
                    })
                if job.assigned_node is None:
                    state_errors.append({
                        "job_id": job.job_id,
                        "issue": "running but no assigned_node",
                    })

            # Check completed jobs
            if job.state == JobState.COMPLETED:
                if job.completion_time is None:
                    state_errors.append({
                        "job_id": job.job_id,
                        "issue": "completed but no completion_time",
                    })

            # Get job metrics
            job_metrics = simulator.metrics_collector.get_job_metrics(job.job_id)
            if job_metrics:
                # Verify state matches
                if job_metrics.completion_time is not None and job.state != JobState.COMPLETED:
                    if job_metrics.completion_time <= simulator.current_time:
                        state_errors.append({
                            "job_id": job.job_id,
                            "issue": f"has completion_time but state is {job.state}",
                        })

        if state_errors:
            affected_job_ids = [e["job_id"] for e in state_errors]
            report.add_violation(ConstraintViolation(
                constraint_name="job_state_consistency",
                violation_type="state_mismatch",
                description=f"{len(state_errors)} jobs have inconsistent states",
                severity="error",
                affected_jobs=affected_job_ids,
            ))

        report.constraint_results["job_state_consistency"]["status"] = (
            "passed" if report.all_passed else "failed"
        )

        return report

    def verify_node_states(self, simulator: Simulator) -> VerificationReport:
        """
        Verify node state consistency.

        Checks:
        - All running nodes have a current_job
        - All idle nodes have no current_job
        - State entry times are valid

        Args:
            simulator: Simulator instance to verify

        Returns:
            VerificationReport with state violations
        """
        report = VerificationReport()
        report.constraint_results["node_state_consistency"] = {
            "checked": True,
            "description": "Verify node states match actual status",
        }

        all_nodes = simulator.fleet_manager.get_all_nodes()
        state_errors = []

        for node in all_nodes:
            # Check running nodes
            if node.state == NodeState.RUNNING:
                if node.current_job_id is None:
                    state_errors.append({
                        "node_id": node.node_id,
                        "issue": "running but no current_job",
                    })
                if node.state_entry_time > simulator.current_time:
                    state_errors.append({
                        "node_id": node.node_id,
                        "issue": "state_entry_time in future",
                    })

            # Check idle/unavailable nodes
            if node.state in [NodeState.IDLE, NodeState.UNAVAILABLE]:
                if node.current_job_id is not None:
                    state_errors.append({
                        "node_id": node.node_id,
                        "issue": f"{node.state} but has current_job",
                    })

        if state_errors:
            affected_node_ids = [e["node_id"] for e in state_errors]
            report.add_violation(ConstraintViolation(
                constraint_name="node_state_consistency",
                violation_type="state_mismatch",
                description=f"{len(state_errors)} nodes have inconsistent states",
                severity="error",
                affected_nodes=affected_node_ids,
            ))

        report.constraint_results["node_state_consistency"]["status"] = (
            "passed" if report.all_passed else "failed"
        )

        return report
