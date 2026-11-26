"""Exercise simulator contracts with real objects and discriminating workloads."""

import importlib.util
import itertools
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cgrsp.config import ResourceType
from cgrsp.data.job_generator import Job as GeneratedJob
from cgrsp.data.job_generator import JobGeneratorConfig, TimeVaryingJobGenerator
from cgrsp.evaluation.runner import ExperimentConfig, ExperimentRunner
from cgrsp.evaluation.verification import AssignmentVerifier, CostVerifier
from cgrsp.schedulers import FIFOScheduler
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.simulator.core import EventType, Simulator
from cgrsp.simulator.queue import Job, JobQueue
from phase3_weight_sensitivity import qos_score
import phase4_monthly as monthly


def test_default_runner_uses_real_simulator():
    config = ExperimentConfig(fleet_config=[(ResourceType.RTX_A5000, 1)],
                              num_episodes=1, simulation_duration=3600,
                              random_seeds=[42])
    result = ExperimentRunner(config).run_scheduler(None, "FIFO")
    assert result.episodes[0].operational_metrics.total_jobs_submitted == 100
    assert result.episodes[0].operational_metrics.total_jobs_completed > 0


def _fifo_run(explicit):
    fleet = RunPodCloudFleetManager(max_concurrency=1, random_seed=42)
    sim = Simulator(fleet_manager=fleet, random_seed=42)
    if explicit:
        sim.set_scheduler(FIFOScheduler(fleet))
    # Block the node until both queued jobs are visible. Their deadlines reverse
    # their submission order, distinguishing FIFO from the queue's EDF ordering.
    for timestamp, deadline in [(0, 1000), (1, 1000), (2, 100)]:
        sim.schedule_event(EventType.JOB_ARRIVAL, timestamp,
                           dict(complexity=1, deadline=deadline, estimated_duration=9999))
    with patch.object(fleet, "request_provisioning", return_value=20) as provisioning:
        with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=50):
            sim.run(300)
    return sim, provisioning.call_count


def test_default_fifo_has_same_provisioning_service_and_order_as_explicit_fifo():
    default, calls = _fifo_run(False)
    explicit, _ = _fifo_run(True)
    assert calls == 3
    assert default.metrics_collector.dispatch_log == explicit.metrics_collector.dispatch_log
    assert [row[0] for row in default.metrics_collector.dispatch_log] == [0, 1, 2]
    assert default.metrics_collector.get_job_metrics(0).start_time == 20
    assert default.metrics_collector.get_job_metrics(0).completion_time == 70


def test_resource_type_helper_filters_real_nodes():
    sim = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1), (ResourceType.RTX_3090, 1)])
    scheduler = FIFOScheduler(sim.fleet_manager)
    assert [node.resource_type for node in scheduler.get_nodes_by_resource_type("RTX_A5000", "idle")] == [ResourceType.RTX_A5000]
    assert scheduler.get_nodes_by_resource_type("RTX_A5000", "running") == []


def test_warmup_excludes_submission_cohort_but_preserves_its_occupancy():
    sim = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1)], random_seed=42)
    sim.submit_job(1, 1000, 9999)
    sim.schedule_event(EventType.JOB_ARRIVAL, 100,
                       dict(complexity=1, deadline=1000, estimated_duration=9999))
    with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=150):
        sim.run(400, warmup_period=100)
    summary = sim.get_metrics_summary()
    assert summary["job_counts"]["submitted"] == summary["job_counts"]["completed"] == 1
    assert summary["timing_metrics"]["avg_waiting_time"] == 50
    assert sim.metrics_collector.get_job_metrics(0).completion_time == 150
    assert [job.job_id for job in sim.metrics_collector.get_completed_job_metrics()] == [1]
    assert all(timestamp >= 100 for timestamp, _ in sim.metrics_collector.get_utilization_time_series())
    report = AssignmentVerifier().verify(sim)
    assert report.all_passed
    assert report.constraint_results["assignment_uniqueness"]["jobs_checked"] == 2
    job_id, node_id, _, start_time = sim.metrics_collector.dispatch_log[1]
    sim.metrics_collector.dispatch_log[1] = (job_id, node_id, 149, start_time)
    overlap = AssignmentVerifier().verify(sim)
    assert any(violation.violation_type == "node_multiple_jobs" for violation in overlap.violations)


def test_empty_queue_accounts_for_lazy_removal():
    queue = JobQueue()
    queue.push(Job(7, 1, 100, 0, 60))
    assert not queue.is_empty()
    queue.remove(7)
    assert queue.is_empty() and len(queue) == 0
    assert queue.pop() is None
    queue.push(Job(7, 1, 200, 1, 60))
    assert queue.pop().job_id == 7


def test_cost_verifier_checks_real_jobs_starting_at_time_zero():
    sim = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1)], random_seed=42)
    sim.submit_job(1, 1000, 9999)
    with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=50):
        sim.run(100)
    job = sim.metrics_collector.get_job_metrics(0)
    assert job.start_time == 0 and job.completion_time == 50
    assert job.execution_cost > 0
    verifier = CostVerifier()
    assert verifier.verify(sim).all_passed
    job.execution_cost += 1
    report = verifier.verify(sim)
    assert not report.all_passed
    assert report.violations[0].violation_type == "cost_mismatch"
    assert report.violations[0].affected_jobs == [0]


def test_queue_history_is_bounded_and_equal_deadline_order_is_retained():
    queue = JobQueue()
    for index in range(5000):
        queue.push(Job(index, 1, 100, 0, 60))
        queue.remove(index)
        queue.get_all_jobs()
    queue.push(Job(6001, 1, 200, 0, 60))
    queue.push(Job(6002, 1, 200, 0, 60))
    assert [job.job_id for job in queue.get_jobs_sorted_by_priority()] == [6001, 6002]
    assert len(queue._queue) < 100


def test_sparse_calibration_restores_state_after_a_failure():
    from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, EXECUTION_TIME_SIGMA_LN, GPUType
    from phase3_sparse_sensitivity import sparse_calibration
    means, sigmas = EXECUTION_TIME_LOOKUP.copy(), EXECUTION_TIME_SIGMA_LN.copy()
    with pytest.raises(RuntimeError, match="interrupted"):
        with sparse_calibration(.75, 2):
            assert EXECUTION_TIME_LOOKUP[(1, GPUType.RTX_A4000)] == pytest.approx(
                .75 * means[(1, GPUType.RTX_A4000)])
            assert EXECUTION_TIME_LOOKUP[(3, GPUType.RTX_A5000)] == means[(3, GPUType.RTX_A5000)]
            raise RuntimeError("interrupted")
    assert EXECUTION_TIME_LOOKUP == means
    assert EXECUTION_TIME_SIGMA_LN == sigmas


def test_time_varying_generator_honours_deadlines_in_trace_and_batch():
    config = JobGeneratorConfig(num_jobs=20, seed=42,
                                deadline_distribution=[(3600, 1)])
    generator = TimeVaryingJobGenerator(config)
    jobs = list(generator.generate())
    jobs.extend(generator.generate_batch(20, 1000))
    assert len(jobs) == 40
    assert len(generator.jobs) == 40
    assert {round(job.deadline - job.submit_time) for job in jobs} == {3600}


@pytest.mark.parametrize("early_rate", [0, .0001])
def test_time_varying_arrivals_follow_integrated_rate_across_a_rate_change(early_rate):
    # Rate 0 before hour one and one job per 100 s afterwards. Over the next
    # 1,000 s the expected count is 10, and the first wait has mean 100 s.
    counts, waits = [], []
    for seed in range(500):
        generator = TimeVaryingJobGenerator(
            JobGeneratorConfig(num_jobs=30, seed=seed),
            arrival_rate_function=lambda timestamp: early_rate if timestamp < 3600 else .01,
            max_arrival_rate=.01)
        jobs = generator.generate()
        if early_rate == 0:
            assert jobs[0].submit_time >= 3600
        counts.append(sum(3600 <= job.submit_time < 4600 for job in jobs))
        waits.append(jobs[0].submit_time - 3600)
    assert 9.5 < np.mean(counts) < 10.5
    if early_rate == 0:
        assert 90 < np.mean(waits) < 110


def test_custom_time_varying_rate_requires_and_respects_a_bound():
    with pytest.raises(ValueError, match="requires max_arrival_rate"):
        TimeVaryingJobGenerator(arrival_rate_function=lambda _: 1)
    generator = TimeVaryingJobGenerator(arrival_rate_function=lambda _: 2,
                                       max_arrival_rate=1)
    with pytest.raises(ValueError, match="between zero"):
        generator.generate()


def test_qos_penalty_uses_seconds_and_per_job_misses():
    assert qos_score(60, .5, alpha2=10) == 65
    assert qos_score(0, 1, alpha2=10) < qos_score(11, 0, alpha2=10)


def test_daily_cutoff_counts_only_arrivals_before_boundary():
    day = monthly.generate_monthly_schedule(3)[4]
    jobs = monthly._day_jobs(day, 4004)
    assert day.n_jobs == 183
    assert len(jobs) == 143
    assert all(job.submit_time < 86400 for job in jobs)


def test_continuous_month_carries_backlog_and_drains_final_cohort(monkeypatch):
    monkeypatch.setattr(monthly, "DURATION", 100)
    monkeypatch.setattr(monthly, "CONCURRENCY", 1)
    days = [monthly.DayConfig(i, i, "normal", 1, .1, [(100, 1)]) for i in range(2)]
    monkeypatch.setattr(monthly, "_day_jobs", lambda day, seed: [
        GeneratedJob(0, 1, 99 if day.day_index == 0 else 1,
                     199 if day.day_index == 0 else 101)])
    factory = lambda fleet, seed, rate: FIFOScheduler(fleet, seed)
    with patch.object(RunPodCloudFleetManager, "request_provisioning", return_value=0):
        with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=150):
            result = monthly.run_month("FIFO", factory, days, 1)
    assert result["total_jobs"] == result["completed"] == 2
    assert result["completion_time_s"] == 399
    assert result["daily_results"][0]["avg_wait_min"] == 0
    assert result["daily_results"][1]["avg_wait_min"] == 148 / 60
    assert result["deadline_missed"] == 2
    assert result["avg_wait_min"] == 74 / 60


def test_drain_finishes_accepted_work_without_admitting_future_arrivals():
    sim = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1)], random_seed=42)
    for timestamp in [0, 150]:
        sim.schedule_event(EventType.JOB_ARRIVAL, timestamp,
                           dict(complexity=1, deadline=1000, estimated_duration=60))
    with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=200):
        sim.run(100, drain=True)
    counts = sim.get_metrics_summary()["job_counts"]
    assert counts["submitted"] == counts["completed"] == 1
    assert sim.metrics_collector.get_job_metrics(0).completion_time == 200


def test_monthly_percentile_is_pooled_instead_of_averaged(monkeypatch):
    monkeypatch.setattr(monthly, "DURATION", 1000)
    monkeypatch.setattr(monthly, "CONCURRENCY", 1)
    days = [monthly.DayConfig(i, i, "normal", 100, .1, [(100, 1)]) for i in range(2)]
    monkeypatch.setattr(monthly, "_day_jobs", lambda day, seed: [
        GeneratedJob(index, 1, index * 3, index * 3 + (100 if day.day_index == 0 else .5))
        for index in range(100)])
    factory = lambda fleet, seed, rate: FIFOScheduler(fleet, seed)
    with patch.object(RunPodCloudFleetManager, "request_provisioning", return_value=0):
        with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", return_value=2):
            result = monthly.run_month("FIFO", factory, days, 1)
    assert result["total_jobs"] == 200
    assert result["daily_results"][0]["p95_tardiness_min"] == 0
    assert result["daily_results"][1]["p95_tardiness_min"] == 1.5 / 60
    assert result["p95_tardiness_min"] == 1.5 / 60
    assert monthly._cohort_metrics([])["p95_tardiness_min"] == 0


def test_figure_cli_reports_unknown_target_and_write_failure(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "thesis/figures/generate_figures.py"
    spec = importlib.util.spec_from_file_location("thesis_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["missing"]) == 1
    def failed_write():
        raise OSError("disk full")
    monkeypatch.setitem(module.FIGURES, "phase3_ci", failed_write)
    assert module.main(["phase3_ci"]) == 1
    assert "disk full" in capsys.readouterr().out


def test_scheduling_diagram_cost_tiebreak_uses_cheapest_complete_assignment(monkeypatch):
    from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, GPUType, PRICING_LOOKUP

    path = Path(__file__).resolve().parents[2] / "thesis/figures/generate_figures.py"
    spec = importlib.util.spec_from_file_location("thesis_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = []
    def capture_labels(fig, name):
        labels.extend(text.get_text() for text in fig.axes[0].texts)
        module.plt.close(fig)
    monkeypatch.setattr(module, "_save", capture_labels)
    module.fig_scheduling_decision()

    assignments = {}
    for label in labels:
        match = re.fullmatch(r'\$j_(\d+) \\to\$ RTX (3090|A5000|A4000)', label)
        if match:
            assignments[int(match.group(1))] = GPUType(f"RTX_{match.group(2)}")
    assert set(assignments) == {1, 2, 3}
    assert len(set(assignments.values())) == 3

    complexity = {1: 1, 2: 2, 3: 1}
    def cost(mapping):
        return sum(EXECUTION_TIME_LOOKUP[(complexity[job], gpu)] * PRICING_LOOKUP[gpu]
                   for job, gpu in mapping.items())
    gpu_types = tuple(assignments.values())
    minimum = min(cost(dict(zip((1, 2, 3), permutation)))
                  for permutation in itertools.permutations(gpu_types))
    assert cost(assignments) == pytest.approx(minimum)


def test_deadline_crossover_guard_accepts_unresolved_midpoint_not_false_tie(tmp_path):
    path = Path(__file__).resolve().parents[2] / "validate_numbers.py"
    spec = importlib.util.spec_from_file_location("thesis_number_validator", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    real_chapter = path.parent / "thesis/chapters/ch1_intro.tex"
    assert not validator._check_deadline_crossover([real_chapter])

    omitted = tmp_path / "omitted.tex"
    omitted.write_text(
        "EDF wins at tight fraction $\\leq 30\\%$ and SPT wins at $\\geq 70\\%$.\n"
    )
    assert validator._check_deadline_crossover([omitted])
    omitted.write_text(
        "EDF wins at tight fraction $\\leq 30\\%$ and SPT wins at $\\geq 70\\%$, "
        "statistically tied at 50%.\n"
    )
    assert validator._check_deadline_crossover([omitted])


def test_kappa_speed_arms_restore_state_and_split_label_from_realised_class():
    import random
    from cgrsp.data.execution_times import EXECUTION_TIME_LOOKUP, GPUType
    import cgrsp.schedulers.rolling_horizon_scheduler as rh
    from cgrsp.simulator.core import Simulator
    from phase3_kappa_speed_sensitivity import ARMS, arm
    means, service_mean, submit = EXECUTION_TIME_LOOKUP.copy(), rh._MEAN_SERVICE_TIME, Simulator.submit_job
    with pytest.raises(RuntimeError, match="interrupted"):
        with arm(ARMS["a4000_scene_matched"], 0):
            assert EXECUTION_TIME_LOOKUP[(3, GPUType.RTX_A4000)] == pytest.approx(
                1.165 * means[(3, GPUType.RTX_A5000)])
            raise RuntimeError("interrupted")
    assert EXECUTION_TIME_LOOKUP == means and rh._MEAN_SERVICE_TIME == service_mean
    with pytest.raises(RuntimeError, match="interrupted"):
        with arm(ARMS["uninformative"], 0):
            assert Simulator.submit_job is not submit
            raise RuntimeError("interrupted")
    assert Simulator.submit_job is submit
    with arm({"label_error": 1.0}, 0):
        sim = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1)], random_seed=0)
        job = sim.submit_job(complexity=1, deadline=100.0, estimated_duration=60.0)
        assert job.true_complexity == 1 and job.complexity == 2
        from cgrsp.data.execution_times import ExecutionTimeModel
        seen = []
        real = ExecutionTimeModel.sample_execution_time

        def spy(complexity, gpu_type, rng, *args, **kwargs):
            seen.append(complexity)
            return real(complexity, gpu_type, rng, *args, **kwargs)

        with patch("cgrsp.data.execution_times.ExecutionTimeModel.sample_execution_time", side_effect=spy):
            sim.run(600, drain=True)
        # Execution follows the true class 1, never the observed label 2.
        assert seen == [1]
    plain = Simulator(fleet_config=[(ResourceType.RTX_A5000, 1)], random_seed=0).submit_job(complexity=1, deadline=100.0, estimated_duration=60.0)
    assert plain.true_complexity is None and plain.complexity == 1
