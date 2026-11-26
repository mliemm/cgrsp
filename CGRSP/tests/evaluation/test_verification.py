"""
Tests for verification module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from cgrsp.evaluation.verification import (
    ConstraintViolation,
    VerificationReport,
    DeadlineVerifier,
    AssignmentVerifier,
    CostVerifier,
    ConstraintVerifier,
    VerificationStatus,
)
from cgrsp.simulator.metrics import JobMetrics


def make_job_metrics(job_id, submit_time, deadline, completion_time, deadline_met):
    """
    Build a completed JobMetrics record with an explicitly chosen deadline_met flag.

    The flag is set directly rather than via calculate_metrics(), so a test can
    record a job whose flag disagrees with its own times, which is precisely the
    inconsistency the verifier exists to catch.
    """
    metrics = JobMetrics(job_id=job_id, submit_time=submit_time, deadline=deadline)
    metrics.start_time = submit_time
    metrics.completion_time = completion_time
    metrics.deadline_met = deadline_met
    return metrics


def make_deadline_simulator(job_metrics_list, reported_misses=None):
    """
    Build a stub simulator exposing only what DeadlineVerifier reads.

    Args:
        job_metrics_list: Completed JobMetrics records
        reported_misses: Fleet-wide miss count; defaults to a count consistent
            with the supplied records, so a test opts in to a mismatch
    """
    if reported_misses is None:
        reported_misses = sum(1 for m in job_metrics_list if m.completion_time > m.deadline)

    simulator = Mock()
    simulator.metrics_collector = Mock()
    simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=job_metrics_list)
    simulator.metrics_collector.system_metrics = Mock(total_jobs_deadline_missed=reported_misses)
    return simulator


@pytest.fixture
def sample_simulator():
    """Create a sample simulator for verification testing."""
    simulator = Mock()

    # Mock job queue
    simulator.job_queue = Mock()
    simulator.job_queue.get_job = Mock(side_effect=lambda job_id: Mock(
        job_id=job_id,
        submit_time=job_id * 1000.0,
        state=Mock(),
        state_entry_time=0.0,
    ))

    simulator.job_queue.get_all_jobs = Mock(return_value=[])

    # Mock metrics collector
    simulator.metrics_collector = Mock()
    simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=[])
    simulator.metrics_collector.get_job_metrics = Mock(return_value=None)
    simulator.metrics_collector.system_metrics = Mock(total_jobs_deadline_missed=0, total_jobs_completed=0)
    simulator.metrics_collector.dispatch_log = []

    # Mock fleet manager
    simulator.fleet_manager = Mock()
    simulator.fleet_manager.get_all_nodes = Mock(return_value=[])
    simulator.fleet_manager.get_node = Mock(return_value=None)
    simulator.current_time = 10000.0

    return simulator


class TestConstraintViolation:
    """Tests for ConstraintViolation dataclass."""

    def test_violation_creation(self):
        """Test violation object creation."""
        violation = ConstraintViolation(
            constraint_name="test_constraint",
            violation_type="test_type",
            description="Test violation",
            severity="error",
            affected_jobs=[1, 2, 3],
            affected_nodes=[10, 20],
        )

        assert violation.constraint_name == "test_constraint"
        assert violation.violation_type == "test_type"
        assert violation.severity == "error"
        assert len(violation.affected_jobs) == 3
        assert len(violation.affected_nodes) == 2

    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = ConstraintViolation(
            constraint_name="test",
            violation_type="type",
            description="desc",
        )

        data = violation.to_dict()

        assert isinstance(data, dict)
        assert data["constraint_name"] == "test"
        assert data["violation_type"] == "type"


class TestVerificationReport:
    """Tests for VerificationReport class."""

    def test_report_creation(self):
        """Test report creation."""
        report = VerificationReport()

        assert report.all_passed is True
        assert report.total_violations == 0
        assert len(report.violations) == 0

    def test_add_violation_error(self):
        """Test adding error violation."""
        report = VerificationReport()

        violation = ConstraintViolation(
            constraint_name="test",
            violation_type="error",
            description="Test error",
            severity="error",
        )

        report.add_violation(violation)

        assert report.all_passed is False
        assert report.total_violations == 1
        assert len(report.violations) == 1

    def test_add_violation_warning(self):
        """Test adding warning violation."""
        report = VerificationReport()

        violation = ConstraintViolation(
            constraint_name="test",
            violation_type="warning",
            description="Test warning",
            severity="warning",
        )

        report.add_violation(violation)

        # Warning doesn't fail the report
        assert report.all_passed is True
        assert report.total_violations == 1

    def test_report_to_dict(self):
        """Test report serialization."""
        report = VerificationReport()

        violation = ConstraintViolation(
            constraint_name="test",
            violation_type="type",
            description="desc",
            severity="error",
        )
        report.add_violation(violation)

        data = report.to_dict()

        assert isinstance(data, dict)
        assert data["all_passed"] is False
        assert data["total_violations"] == 1
        assert len(data["violations"]) == 1


class TestDeadlineVerifier:
    """Tests for DeadlineVerifier class."""

    def test_verify_no_jobs(self, sample_simulator):
        """Test verification with no completed jobs."""
        verifier = DeadlineVerifier()
        report = verifier.verify(sample_simulator)

        assert report.all_passed is True
        assert report.total_violations == 0

    def test_verify_with_correctly_flagged_jobs(self):
        """A consistent record of one on-time and one late job raises nothing."""
        verifier = DeadlineVerifier()

        simulator = make_deadline_simulator([
            # Submitted at 1000, due at 6000, finished at 5000: on time
            make_job_metrics(1, 1000.0, 6000.0, 5000.0, deadline_met=True),
            # Submitted at 2000, due at 7000, finished at 7500: late, marked late
            make_job_metrics(2, 2000.0, 7000.0, 7500.0, deadline_met=False),
        ])

        report = verifier.verify(simulator)

        assert report.all_passed is True
        assert report.total_violations == 0

    def test_verify_detects_missed_deadline_not_marked(self):
        """A job finishing past its own deadline while flagged as met is a violation."""
        verifier = DeadlineVerifier()

        # Due at 7000, finished at 9000, yet recorded as having met its deadline.
        late_job = make_job_metrics(7, 2000.0, 7000.0, 9000.0, deadline_met=True)
        # The fleet-wide counter agrees with the (wrong) flag, so the recount is
        # what has to catch this, not the counter.
        simulator = make_deadline_simulator([late_job], reported_misses=1)

        report = verifier.verify(simulator)

        assert report.all_passed is False
        assert report.total_violations == 1
        violation = report.violations[0]
        assert violation.violation_type == "missed_deadline_not_marked"
        assert violation.severity == "error"
        assert violation.affected_jobs == [7]

    def test_verify_honours_per_job_deadlines(self):
        """
        Heterogeneous deadlines are each judged against their own d_j.

        The main experiments mix a 1 h tight window with an 8 h loose one, so a
        verifier holding every job to a single threshold would report a spurious
        violation on one of these two correct records.
        """
        verifier = DeadlineVerifier()

        simulator = make_deadline_simulator([
            # Tight 1 h window, finished in 30 min: on time
            make_job_metrics(1, 0.0, 3600.0, 1800.0, deadline_met=True),
            # Loose 8 h window, finished in 5 h: on time
            make_job_metrics(2, 0.0, 28800.0, 18000.0, deadline_met=True),
        ])

        report = verifier.verify(simulator)

        assert report.all_passed is True
        assert report.total_violations == 0

    def test_verify_detects_miss_count_mismatch(self):
        """The reported miss count must agree with a recount over the job records."""
        verifier = DeadlineVerifier()

        # One genuine, correctly flagged miss, but the fleet-wide counter says three.
        simulator = make_deadline_simulator(
            [make_job_metrics(1, 0.0, 3600.0, 4000.0, deadline_met=False)],
            reported_misses=3,
        )

        report = verifier.verify(simulator)

        assert report.all_passed is False
        assert any(v.violation_type == "miss_count_mismatch" for v in report.violations)

    def test_verify_flags_on_time_job_marked_as_missed(self):
        """An on-time job flagged as missed is reported, as a warning only."""
        verifier = DeadlineVerifier()

        simulator = make_deadline_simulator(
            [make_job_metrics(4, 0.0, 3600.0, 1200.0, deadline_met=False)],
            reported_misses=0,
        )

        report = verifier.verify(simulator)

        # A warning does not fail the report, but it must still be recorded.
        assert report.total_violations == 1
        assert report.violations[0].violation_type == "incorrect_deadline_flag"
        assert report.violations[0].severity == "warning"
        assert report.all_passed is True


class TestAssignmentVerifier:
    """Tests for AssignmentVerifier class."""

    def test_verify_no_nodes(self, sample_simulator):
        """Test verification with no nodes."""
        verifier = AssignmentVerifier()
        report = verifier.verify(sample_simulator)

        assert report.all_passed is True

    def test_verify_proper_assignments(self):
        """Test verification with proper assignments."""
        from cgrsp.simulator.fleet import NodeState

        verifier = AssignmentVerifier()

        simulator = Mock()
        simulator.fleet_manager = Mock()
        simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=[])
        simulator.metrics_collector.dispatch_log = []
        simulator.metrics_collector.system_metrics.total_jobs_completed = 0

        # Create mock nodes with proper assignments
        node1 = Mock()
        node1.node_id = 1
        node1.state = NodeState.RUNNING
        node1.current_job_id = 100

        node2 = Mock()
        node2.node_id = 2
        node2.state = NodeState.IDLE
        node2.current_job_id = None

        simulator.fleet_manager.get_all_nodes = Mock(return_value=[node1, node2])

        report = verifier.verify(simulator)

        assert report.all_passed is True

    def test_verify_single_assignment_passes(self):
        """A single running node holding one job raises nothing."""
        from cgrsp.simulator.fleet import NodeState

        verifier = AssignmentVerifier()

        simulator = Mock()
        simulator.fleet_manager = Mock()
        simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=[])
        simulator.metrics_collector.dispatch_log = []
        simulator.metrics_collector.system_metrics.total_jobs_completed = 0

        node1 = Mock()
        node1.node_id = 1
        node1.state = NodeState.RUNNING
        node1.current_job_id = 100

        simulator.fleet_manager.get_all_nodes = Mock(return_value=[node1])

        report = verifier.verify(simulator)

        assert report.all_passed is True

    def test_verify_detects_job_on_multiple_nodes(self):
        """One job running on two nodes at once is a violation."""
        from cgrsp.simulator.fleet import NodeState

        verifier = AssignmentVerifier()

        simulator = Mock()
        simulator.fleet_manager = Mock()
        simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=[])
        simulator.metrics_collector.dispatch_log = []
        simulator.metrics_collector.system_metrics.total_jobs_completed = 0

        node1 = Mock()
        node1.node_id = 1
        node1.state = NodeState.RUNNING
        node1.current_job_id = 100

        node2 = Mock()
        node2.node_id = 2
        node2.state = NodeState.RUNNING
        node2.current_job_id = 100  # same job, second node

        simulator.fleet_manager.get_all_nodes = Mock(return_value=[node1, node2])

        report = verifier.verify(simulator)

        assert report.all_passed is False
        assert report.total_violations == 1
        violation = report.violations[0]
        assert violation.violation_type == "job_multiple_nodes"
        assert violation.affected_jobs == [100]
        assert sorted(violation.affected_nodes) == [1, 2]

    @staticmethod
    def _history_simulator(runs):
        """Drive a real MetricsCollector: runs are (job, node, dispatch, start, completion)."""
        from cgrsp.simulator.metrics import MetricsCollector
        mc = MetricsCollector()
        for j, node, d, s0, e0 in runs:
            mc.record_job_submission(j, 0.0, 1e6)
        for j, node, d, s0, e0 in runs:
            mc.record_job_start(j, s0, node, "RTX_3090", dispatch_time=d)
        for j, node, d, s0, e0 in sorted(runs, key=lambda r: r[4]):
            mc.record_job_completion(j, e0, 1.0)
        simulator = Mock()
        simulator.fleet_manager.get_all_nodes = Mock(return_value=[])
        simulator.metrics_collector = mc
        return simulator

    def test_history_back_to_back_jobs_pass(self):
        """Jobs that follow each other on one node, and a parallel job on another, pass."""
        report = AssignmentVerifier().verify(self._history_simulator(
            [(1, 1, 0.0, 0.0, 10.0), (2, 1, 10.0, 12.0, 20.0), (3, 2, 5.0, 5.0, 15.0)]))
        assert report.all_passed is True
        assert report.constraint_results["assignment_uniqueness"]["jobs_checked"] == 3

    def test_history_detects_overlap_on_one_node(self):
        """Two jobs overlapping in time on the same node is a violation even after the run ends."""
        report = AssignmentVerifier().verify(self._history_simulator(
            [(1, 1, 0.0, 0.0, 10.0), (2, 1, 5.0, 5.0, 15.0)]))
        assert report.all_passed is False
        assert any(v.violation_type == "node_multiple_jobs" for v in report.violations)

    def test_history_detects_overlap_during_provisioning(self):
        """A slot is held from dispatch: a job run inside another job's provisioning wait overlaps."""
        report = AssignmentVerifier().verify(self._history_simulator(
            [(1, 1, 0.0, 120.0, 300.0), (2, 1, 10.0, 10.0, 100.0)]))
        assert report.all_passed is False
        assert any(v.violation_type == "node_multiple_jobs" and sorted(v.affected_jobs) == [1, 2]
                   for v in report.violations)

    def test_history_detects_job_dispatched_twice(self):
        """A job dispatched to a second node keeps one per-job record; the dispatch log exposes it."""
        simulator = self._history_simulator([(1, 1, 0.0, 0.0, 10.0), (2, 2, 0.0, 0.0, 10.0)])
        simulator.metrics_collector.record_job_start(2, 20.0, 3, "RTX_3090", dispatch_time=20.0)
        report = AssignmentVerifier().verify(simulator)
        assert report.all_passed is False
        assert any(v.violation_type == "job_multiple_nodes" and v.affected_jobs == [2]
                   and v.affected_nodes == [2, 3] for v in report.violations)

    def test_history_detects_completion_recorded_twice(self):
        """A second completion of the same job is caught by the completion-count cross-check."""
        simulator = self._history_simulator([(1, 1, 0.0, 0.0, 10.0)])
        simulator.metrics_collector.record_job_completion(1, 11.0, 1.0)
        report = AssignmentVerifier().verify(simulator)
        assert report.all_passed is False
        assert any(v.violation_type == "completion_count_mismatch" for v in report.violations)


class TestCostVerifier:
    """Tests for CostVerifier class."""

    def test_verify_no_jobs(self, sample_simulator):
        """Test verification with no completed jobs."""
        verifier = CostVerifier()
        report = verifier.verify(sample_simulator)

        assert report.all_passed is True

    @staticmethod
    def _cost_simulator(execution_cost, hourly_rate=36.0):
        """Stub a simulator with one completed job run for 1000 s at hourly_rate."""
        job_metrics = Mock(
            job_id=1,
            assigned_node_id=10,
            resource_type="RTX_3090",
            start_time=1000.0,
            completion_time=2000.0,  # 1000 seconds = 1000/3600 hours
            execution_cost=execution_cost,
        )

        node = Mock()
        node.get_hourly_cost = Mock(return_value=hourly_rate)

        simulator = Mock()
        simulator.metrics_collector = Mock()
        simulator.metrics_collector.get_completed_job_metrics = Mock(return_value=[job_metrics])
        simulator.fleet_manager = Mock()
        simulator.fleet_manager.get_node = Mock(return_value=node)
        return simulator

    def test_verify_cost_calculation(self):
        """A cost matching execution time x hourly rate raises nothing."""
        verifier = CostVerifier()

        # 1000 s = 0.2778 h at $36/h = $10.00
        report = verifier.verify(self._cost_simulator(execution_cost=10.0))

        assert report.all_passed is True
        assert report.total_violations == 0

    def test_verify_detects_cost_mismatch(self):
        """A recorded cost that does not match the node's rate is a violation."""
        verifier = CostVerifier()

        # Expected $10.00 for the same run, but $25.00 was recorded.
        report = verifier.verify(self._cost_simulator(execution_cost=25.0))

        assert report.all_passed is False
        assert report.total_violations == 1
        violation = report.violations[0]
        assert violation.violation_type == "cost_mismatch"
        assert violation.affected_jobs == [1]


class TestConstraintVerifier:
    """Tests for ConstraintVerifier class."""

    def test_initialization(self):
        """Test verifier initialization."""
        verifier = ConstraintVerifier()
        assert isinstance(verifier.deadline_verifier, DeadlineVerifier)
        assert isinstance(verifier.assignment_verifier, AssignmentVerifier)
        assert isinstance(verifier.cost_verifier, CostVerifier)

    def test_verify_all(self, sample_simulator):
        """Test running all verifications."""
        verifier = ConstraintVerifier()
        report = verifier.verify_all(sample_simulator)

        assert isinstance(report, VerificationReport)
        assert "deadline_enforcement" in report.constraint_results
        assert "assignment_uniqueness" in report.constraint_results
        assert "cost_calculation" in report.constraint_results
        assert report.all_passed is True
        assert report.total_violations == 0

    def test_verify_all_surfaces_a_violation(self, sample_simulator):
        """A violation found by one sub-verifier reaches the combined report."""
        # Due at 3600, finished at 9000, yet recorded as having met its deadline.
        sample_simulator.metrics_collector.get_completed_job_metrics = Mock(
            return_value=[make_job_metrics(3, 0.0, 3600.0, 9000.0, deadline_met=True)]
        )
        sample_simulator.metrics_collector.system_metrics = Mock(total_jobs_deadline_missed=1, total_jobs_completed=1)
        sample_simulator.metrics_collector.dispatch_log = [(3, 1, 0.0, 0.0)]

        report = ConstraintVerifier().verify_all(sample_simulator)

        assert report.all_passed is False
        assert report.total_violations == 1
        assert report.violations[0].violation_type == "missed_deadline_not_marked"

    def test_verify_job_states(self, sample_simulator):
        """Test job state verification."""
        from cgrsp.simulator.queue import JobState

        verifier = ConstraintVerifier()

        # Add mock jobs
        job1 = Mock()
        job1.job_id = 1
        job1.state = JobState.RUNNING
        job1.start_time = 1000.0
        job1.assigned_node = 10

        job2 = Mock()
        job2.job_id = 2
        job2.state = JobState.COMPLETED
        job2.completion_time = 2000.0

        sample_simulator.job_queue.get_all_jobs = Mock(return_value=[job1, job2])
        sample_simulator.metrics_collector.get_job_metrics = Mock(return_value=None)

        report = verifier.verify_job_states(sample_simulator)

        assert isinstance(report, VerificationReport)
        assert "job_state_consistency" in report.constraint_results

    def test_verify_node_states(self, sample_simulator):
        """Test node state verification."""
        from cgrsp.simulator.fleet import NodeState

        verifier = ConstraintVerifier()

        # Add mock nodes
        node1 = Mock()
        node1.node_id = 1
        node1.state = NodeState.RUNNING
        node1.current_job_id = 100
        node1.state_entry_time = 1000.0

        node2 = Mock()
        node2.node_id = 2
        node2.state = NodeState.IDLE
        node2.current_job_id = None

        sample_simulator.fleet_manager.get_all_nodes = Mock(return_value=[node1, node2])

        report = verifier.verify_node_states(sample_simulator)

        assert isinstance(report, VerificationReport)
        assert "node_state_consistency" in report.constraint_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_simulated_run_logs_dispatch_before_provisioned_start():
    """An end-to-end run logs each job's dispatch instant, so occupancy includes the provisioning wait."""
    from cgrsp.config import ResourceType
    from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
    from cgrsp.schedulers import FIFOScheduler
    from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
    from cgrsp.simulator.core import EventType, Simulator

    fleet = RunPodCloudFleetManager(max_concurrency=3, supported_types=[ResourceType.RTX_3090], random_seed=7)
    sim = Simulator(fleet_manager=fleet, random_seed=7)
    sim.set_scheduler(FIFOScheduler(fleet_manager=fleet, seed=7))
    jobs = JobGenerator(JobGeneratorConfig(num_jobs=30, arrival_rate=0.05, seed=7)).generate()
    for job in jobs:
        sim.schedule_event(EventType.JOB_ARRIVAL, job.submit_time, {
            "complexity": job.complexity, "deadline": job.deadline, "estimated_duration": 100.0})
    sim.run(duration=86400.0)

    log = sim.metrics_collector.dispatch_log
    assert len(log) == 30
    assert any(t_dispatch < t_start for _, _, t_dispatch, t_start in log)
    report = AssignmentVerifier().verify(sim)
    assert report.all_passed is True
    assert report.constraint_results["assignment_uniqueness"]["jobs_checked"] == 30
