"""Fleet-composition sensitivity sweep: mix table, anchor equivalence, ordering flag.

The sweep itself takes minutes, so these tests pin the properties the sweep
depends on instead of running it: the declared compositions, the anchor's slot
mapping (which is what makes it reproduce statistical.json), the reduced-fleet
dispatch path, and the derivation of the ordering flag.
"""
import copy
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import phase3_fleet_mix as fm
from cgrsp.config import ResourceType
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager

R = ResourceType

EXPECTED_COUNTS = {
    "all-types": Counter({R.RTX_3090: 2, R.RTX_A5000: 1,
                          R.RTX_A4500: 1, R.RTX_A4000: 1}),
    "3090-heavy": Counter({R.RTX_3090: 3, R.RTX_A5000: 1, R.RTX_A4000: 1}),
    "A5000-heavy": Counter({R.RTX_A5000: 3, R.RTX_3090: 1, R.RTX_A4000: 1}),
    "A4000-heavy": Counter({R.RTX_A4000: 3, R.RTX_3090: 1, R.RTX_A5000: 1}),
    "3090-only": Counter({R.RTX_3090: 5}),
    "A5000-only": Counter({R.RTX_A5000: 5}),
}


def _node_types(fleet):
    return [n.resource_type for n in fleet.get_all_nodes()]


# --- mix table --------------------------------------------------------------

def test_every_mix_fills_five_slots():
    assert fm.CONCURRENCY == 5
    assert set(fm.MIXES) == set(EXPECTED_COUNTS)
    for mix, slots in fm.MIXES.items():
        assert len(slots) == 5, mix
        assert all(isinstance(s, ResourceType) for s in slots), mix


def test_mix_slot_counts_match_declared_composition():
    for mix, expected in EXPECTED_COUNTS.items():
        assert Counter(fm.MIXES[mix]) == expected, mix


def test_anchor_is_the_default_round_robin():
    # The anchor table is the round-robin expansion of the default type list.
    default_types = ResourceType.supported_gpus()
    assert fm.MIXES[fm.ANCHOR_MIX] == [
        default_types[i % len(default_types)] for i in range(5)
    ]
    # And building it maps slot ids to the same types as the default manager.
    anchor = fm.build_fleet(fm.MIXES[fm.ANCHOR_MIX], 0)
    default = RunPodCloudFleetManager(
        max_concurrency=5, supported_types=default_types, random_seed=0)
    assert [(n.node_id, n.resource_type) for n in anchor.get_all_nodes()] == \
           [(n.node_id, n.resource_type) for n in default.get_all_nodes()]


def test_status_stream_is_held_canonical_and_slots_keep_the_mix():
    assert fm.CANONICAL_TYPES == ResourceType.supported_gpus()
    for mix, slots in fm.MIXES.items():
        fleet = fm.build_fleet(slots, 0)
        assert _node_types(fleet) == list(slots), mix
        assert fleet.supported_types == list(fm.CANONICAL_TYPES), mix


def test_reduced_mix_has_zero_slots_of_omitted_types():
    omitted = {
        "3090-only": (R.RTX_A5000, R.RTX_A4500, R.RTX_A4000),
        "A5000-only": (R.RTX_3090, R.RTX_A4500, R.RTX_A4000),
    }
    for mix, dropped in omitted.items():
        slots = fm.MIXES[mix]
        types = _node_types(fm.build_fleet(slots, 0))
        assert len(types) == 5
        for resource_type in dropped:
            assert slots.count(resource_type) == 0, mix
            assert types.count(resource_type) == 0, mix


# --- dispatch path ----------------------------------------------------------

def test_no_focal_scheduler_crashes_on_a_reduced_fleet():
    slots = fm.MIXES["3090-only"]
    for policy in fm.FOCAL_POLICIES:
        run = fm.run_one(fm.SCHEDULER_MAP[policy], 0, slots, n_jobs=40)
        assert run["completed"] == 40, policy
        assert 0.0 <= run["deadline_miss_pct"] <= 100.0, policy
        assert math.isfinite(run["avg_wait_min"]) and run["avg_wait_min"] >= 0.0, policy
        assert math.isfinite(run["total_cost"]) and run["total_cost"] > 0.0, policy


def test_anchor_reproduces_statistical_json_for_fifo_seed_zero():
    reference = json.loads(fm.STATISTICAL_JSON.read_text())
    run = fm.run_one(fm.SCHEDULER_MAP["FIFO"], 0, fm.MIXES[fm.ANCHOR_MIX])
    for raw_key, run_key, _, ndigits in fm.RAW_FIELDS:
        assert round(float(run[run_key]), ndigits) == \
               round(float(reference["schedulers"]["FIFO"][raw_key][0]), ndigits), raw_key


# --- ordering flag ----------------------------------------------------------

BASE_MISS = {
    "mix-a": {"FIFO": 20.0, "SPT": 18.0, "EDF": 15.0,
              "SPT+Rescue": 14.0, "RH": 10.0, "CADR": 11.0},
    "mix-b": {"FIFO": 25.0, "SPT": 22.0, "EDF": 19.0,
              "SPT+Rescue": 17.0, "RH": 12.0, "CADR": 13.0},
}


def test_ordering_stable_true_for_a_consistent_ordering():
    checks = fm.ordering_checks(BASE_MISS)
    assert checks["ordering_stable"] is True
    for mix, per_mix in checks["per_mix"].items():
        assert per_mix["two_lowest_miss"] == ["RH", "CADR"], mix
        assert all(per_mix["deadline_aware_below_fifo"].values()), mix


def test_ordering_stable_flips_when_input_ordering_is_reversed():
    # Reversing the miss ordering (negating every mean) makes FIFO and SPT the
    # two lowest and puts the deadline-aware policies above FIFO.
    reversed_miss = {
        mix: {policy: -value for policy, value in miss.items()}
        for mix, miss in BASE_MISS.items()
    }
    assert fm.ordering_checks(BASE_MISS)["ordering_stable"] is True
    checks = fm.ordering_checks(reversed_miss)
    assert checks["ordering_stable"] is False
    assert checks["per_mix"]["mix-a"]["two_lowest_miss"] == ["FIFO", "SPT"]


def test_ordering_stable_requires_deadline_aware_policies_below_fifo():
    miss = copy.deepcopy(BASE_MISS)
    # RH and CADR stay the two lowest, but EDF now sits above FIFO.
    miss["mix-b"]["EDF"] = 30.0
    checks = fm.ordering_checks(miss)
    assert checks["per_mix"]["mix-b"]["rh_cadr_two_lowest"] is True
    assert checks["per_mix"]["mix-b"]["deadline_aware_below_fifo"]["EDF"] is False
    assert checks["ordering_stable"] is False


def test_ordering_stable_requires_rh_and_cadr_to_lead():
    miss = copy.deepcopy(BASE_MISS)
    # Every deadline-aware policy is below FIFO, but SPT takes the second place.
    miss["mix-a"]["SPT"] = 5.0
    checks = fm.ordering_checks(miss)
    assert checks["per_mix"]["mix-a"]["rh_cadr_two_lowest"] is False
    assert all(checks["per_mix"]["mix-a"]["deadline_aware_below_fifo"].values())
    assert checks["ordering_stable"] is False


def test_ordering_checks_ignore_policy_insertion_order():
    reordered = {
        mix: dict(reversed(list(miss.items())))
        for mix, miss in BASE_MISS.items()
    }
    assert fm.ordering_checks(reordered) == fm.ordering_checks(BASE_MISS)


def test_ordering_checks_reject_a_missing_policy():
    miss = copy.deepcopy(BASE_MISS)
    del miss["mix-a"]["CADR"]
    try:
        fm.ordering_checks(miss)
    except KeyError as exc:
        assert "CADR" in str(exc)
    else:
        raise AssertionError("missing policy was not rejected")


# --- anchor equivalence checker --------------------------------------------

def _write_reference(tmp_path, values_by_policy, n_seeds=2):
    schedulers = {}
    for policy, fields in values_by_policy.items():
        schedulers[policy] = {
            raw_key: list(fields[raw_key])[:n_seeds]
            for raw_key, _, _, _ in fm.RAW_FIELDS
        }
    path = tmp_path / "statistical.json"
    path.write_text(json.dumps({"schedulers": schedulers}))
    return path


def _raw_by_policy(policies, n_seeds=2):
    raw = {}
    for index, policy in enumerate(policies):
        raw[policy] = {
            raw_key: [10.0 * (index + 1) + seed for seed in range(n_seeds)]
            for raw_key, _, _, _ in fm.RAW_FIELDS
        }
    return raw


def test_anchor_equivalence_checker_accepts_equal_values(tmp_path):
    raw = _raw_by_policy(("FIFO", "SPT"))
    path = _write_reference(tmp_path, raw)
    result = fm.check_anchor_equivalence(
        raw, 2, ("FIFO", "SPT"), reference_path=path)
    assert result["passed"] is True
    assert result["mismatch_count"] == 0
    assert result["per_policy"]["SPT"]["compared_values"] == 2 * len(fm.RAW_FIELDS)


def test_anchor_equivalence_checker_detects_a_mismatch(tmp_path):
    raw = _raw_by_policy(("FIFO",))
    path = _write_reference(tmp_path, raw)
    raw["FIFO"]["raw_miss_pct"][1] += 0.001  # above the four-decimal comparison
    result = fm.check_anchor_equivalence(
        raw, 2, ("FIFO",), reference_path=path)
    assert result["passed"] is False
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["field"] == "raw_miss_pct"
    assert result["mismatches"][0]["seed"] == 1


def test_anchor_equivalence_checker_rejects_a_short_reference(tmp_path):
    raw = _raw_by_policy(("FIFO",))
    path = _write_reference(tmp_path, raw, n_seeds=1)
    try:
        fm.check_anchor_equivalence(raw, 2, ("FIFO",), reference_path=path)
    except ValueError as exc:
        assert "need 2" in str(exc)
    else:
        raise AssertionError("short reference was not rejected")


def test_sweep_rejects_a_single_seed():
    try:
        fm.run_fleet_mix_sweep(n_seeds=1, policies=("FIFO",), verbose=False)
    except ValueError as exc:
        assert "two seeds" in str(exc)
    else:
        raise AssertionError("a single seed was not rejected")


# --- determinism digest -----------------------------------------------------

def test_normalized_digest_ignores_timing_and_itself():
    base = {
        "metadata": {"n_seeds": 2, "elapsed_sec": 1.5},
        "mixes": {"all-types": {"policies": {"FIFO": {"raw_wait_min": [1.0, 2.0]}}}},
    }
    slowed = copy.deepcopy(base)
    slowed["metadata"]["elapsed_sec"] = 999.0
    with_digest = copy.deepcopy(base)
    with_digest["metadata"]["normalized_sha256"] = "0" * 64

    assert fm.normalized_digest(base) == fm.normalized_digest(slowed)
    assert fm.normalized_digest(base) == fm.normalized_digest(with_digest)

    changed = copy.deepcopy(base)
    changed["mixes"]["all-types"]["policies"]["FIFO"]["raw_wait_min"][1] = 2.5
    assert fm.normalized_digest(base) != fm.normalized_digest(changed)
