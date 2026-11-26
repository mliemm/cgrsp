"""
Phase 3 - Fleet-Composition Sensitivity
=======================================
Runs the same hectic-day workload on six five-slot fleets that differ only in
GPU composition, and pairs every mix with the all-types anchor on the same
seeds. The workload (job count, arrival rate, deadline mix, duration), the
scheduler set and the statistic implementations are imported from
phase3_statistical so the sweep cannot drift from the canonical Phase 3 setup.

Mix table (five slots each, max_concurrency=5):

  mix          slots (by slot id 0..4)
  -----------  ------------------------------------------------------------
  all-types    RTX_3090, RTX_A5000, RTX_A4500, RTX_A4000, RTX_3090   anchor
  3090-heavy   RTX_3090, RTX_3090,  RTX_3090,  RTX_A5000, RTX_A4000
  A5000-heavy  RTX_A5000, RTX_A5000, RTX_A5000, RTX_3090,  RTX_A4000
  A4000-heavy  RTX_A4000, RTX_A4000, RTX_A4000, RTX_3090,  RTX_A5000
  3090-only    RTX_3090 x 5
  A5000-only   RTX_A5000 x 5

The anchor entry is the round-robin expansion of ResourceType.supported_gpus()
over five slots: two RTX 3090 plus one each of A5000, A4500 and A4000. Listing
it in round-robin order keeps slot ids on the default type mapping, so the
anchor run reproduces Outputs/phase3/statistical.json exactly (checked for the
shared policies on every seed of the sweep, see check_anchor_equivalence).

Availability stream. The slot list is passed as supported_types to
RunPodCloudFleetManager, whose round robin assigns one type per slot id.
update_stock_statuses then draws one stock status per entry of supported_types
on every provisioning request, so a five-entry list containing a duplicated
type would consume a different number of draws than the default four-type run
and shift the availability stream. build_fleet therefore restores
supported_types to the canonical four types after construction. Every mix then
observes the same availability status draw stream; what differs between mixes
is the slot composition and the dispatch sequence it induces.

Policies and seeds. Ten seeds (0-9, matching the existing Phase 3 tuning
sweeps) on the six focal policies (FIFO, SPT, EDF, SPT+Rescue, RH, CADR). The
ten-policy set measured 9.5 s per seed, which projects to about 9.5 minutes
over six mixes and ten seeds, above the eight-minute budget; it remains
available via --policies all.

Output: Outputs/phase3/fleet_mix.json
"""

import _repro  # noqa: F401 -- pins PYTHONHASHSEED=0 for reproducible runs
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgrsp.config import ResourceType
from cgrsp.data.job_generator import JobGenerator, JobGeneratorConfig
from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
from cgrsp.simulator.core import EventType, Simulator
from phase3_statistical import (
    ARRIVAL_RATE,
    CONCURRENCY,
    DEADLINE_DIST,
    DURATION,
    N_JOBS,
    SCHEDULER_MAP,
    _ci95,
    _paired_t,
)

N_SEEDS = 10
ANCHOR_MIX = "all-types"

# The canonical availability-model type list. Held constant across mixes; see
# build_fleet and the module docstring.
CANONICAL_TYPES = list(ResourceType.supported_gpus())

MIXES = {
    "all-types": [
        ResourceType.RTX_3090,
        ResourceType.RTX_A5000,
        ResourceType.RTX_A4500,
        ResourceType.RTX_A4000,
        ResourceType.RTX_3090,
    ],
    "3090-heavy": [
        ResourceType.RTX_3090,
        ResourceType.RTX_3090,
        ResourceType.RTX_3090,
        ResourceType.RTX_A5000,
        ResourceType.RTX_A4000,
    ],
    "A5000-heavy": [
        ResourceType.RTX_A5000,
        ResourceType.RTX_A5000,
        ResourceType.RTX_A5000,
        ResourceType.RTX_3090,
        ResourceType.RTX_A4000,
    ],
    "A4000-heavy": [
        ResourceType.RTX_A4000,
        ResourceType.RTX_A4000,
        ResourceType.RTX_A4000,
        ResourceType.RTX_3090,
        ResourceType.RTX_A5000,
    ],
    "3090-only": [ResourceType.RTX_3090] * 5,
    "A5000-only": [ResourceType.RTX_A5000] * 5,
}

FOCAL_POLICIES = ("FIFO", "SPT", "EDF", "SPT+Rescue", "RH", "CADR")
DEADLINE_AWARE = ("EDF", "SPT+Rescue", "RH", "CADR")

# Output key, per-run metric key, aggregate key, decimals used for the exact
# anchor comparison. The output key names match statistical.json.
RAW_FIELDS = (
    ("raw_wait_min", "avg_wait_min", "wait_min", 4),
    ("raw_miss_pct", "deadline_miss_pct", "miss_pct", 4),
    ("raw_cost_usd", "total_cost", "cost_usd", 6),
    ("raw_util_pct", "utilization_pct", "util_pct", 4),
    ("raw_avg_tardiness_min", "avg_tardiness_min", "avg_tardiness_min", 6),
    ("raw_p95_tardiness_min", "p95_tardiness_min", "p95_tardiness_min", 6),
)

STATISTICAL_JSON = Path(__file__).parent / "Outputs/phase3/statistical.json"
OUTPUT_JSON = Path(__file__).parent / "Outputs/phase3/fleet_mix.json"

# Only these metadata fields change between two runs of the same seed set.
VOLATILE_METADATA_KEYS = ("elapsed_sec", "normalized_sha256")


# --- fleet construction and one simulation run ------------------------------

def build_fleet(slots, seed):
    """Build the cloud fleet for one mix.

    ``slots`` is the mix list, passed as supported_types so the round robin
    gives slot id i the type ``slots[i]``. supported_types is then restored to
    the canonical four-type list: after construction its only reader is
    update_stock_statuses, which draws one status per entry on every
    provisioning request, so iterating a five-entry list with a duplicate type
    would desynchronise the status stream from the default run. Holding it
    canonical keeps the availability stream identical across mixes and makes
    the anchor bit-identical to the manager phase3_statistical builds.
    """
    cloud = RunPodCloudFleetManager(
        max_concurrency=len(slots),
        supported_types=list(slots),
        random_seed=seed,
    )
    cloud.supported_types = list(CANONICAL_TYPES)
    return cloud


def run_one(scheduler_cls, seed, slots, n_jobs=N_JOBS):
    """Run one scheduler on one mix and seed; return the six Phase 3 metrics."""
    job_config = JobGeneratorConfig(
        num_jobs=n_jobs, arrival_rate=ARRIVAL_RATE, seed=seed,
        deadline_distribution=DEADLINE_DIST,
    )
    jobs = JobGenerator(job_config).generate()

    cloud = build_fleet(slots, seed)
    sim = Simulator(fleet_manager=cloud, random_seed=seed)
    sched = scheduler_cls(fleet_manager=cloud, seed=seed)
    sim.set_scheduler(sched)

    for job in jobs:
        sim.schedule_event(
            EventType.JOB_ARRIVAL,
            job.submit_time,
            {
                "complexity": job.complexity,
                "deadline": job.deadline,
                "estimated_duration": job.estimated_duration or 100.0,
            },
        )

    sim.run(duration=DURATION)
    m = sim.get_metrics_summary()
    return {
        "completed": m["job_counts"]["completed"],
        "avg_wait_min": round(m["timing_metrics"]["avg_waiting_time"] / 60.0, 4),
        "deadline_miss_pct": round(m["job_counts"]["deadline_miss_rate_pct"], 4),
        "total_cost": round(m["cost_metrics"]["total_cost"], 6),
        "utilization_pct": round(m["utilization"]["avg_pct"], 4),
        "avg_tardiness_min": m.get("avg_tardiness_s", 0.0) / 60.0,
        "p95_tardiness_min": m.get("p95_tardiness_s", 0.0) / 60.0,
    }


def aggregate_runs(runs):
    """Per-seed raw arrays plus 95% CI blocks, keyed as in statistical.json."""
    out = {}
    for raw_key, run_key, agg_key, _ in RAW_FIELDS:
        values = [r[run_key] for r in runs]
        out[raw_key] = values
        out[agg_key] = _ci95(values)
    return out


# --- anchor equivalence -----------------------------------------------------

def check_anchor_equivalence(raw_by_policy, n_seeds, policies, reference_path=STATISTICAL_JSON):
    """Compare the anchor mix with statistical.json seed by seed.

    ``raw_by_policy`` is {policy: {raw_key: [per-seed values]}}. Every shared
    policy and every raw metric must match the reference value at the stored
    number of decimals; the tardiness fields are stored unrounded in the
    reference, so they are compared at six decimals (minutes).
    """
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text())
    try:
        reference_label = str(reference_path.relative_to(Path(__file__).parent))
    except ValueError:
        reference_label = str(reference_path)
    per_policy = {}
    mismatches = []
    for policy in policies:
        policy_mismatches = 0
        for raw_key, _, _, ndigits in RAW_FIELDS:
            ref_values = reference["schedulers"][policy][raw_key]
            if len(ref_values) < n_seeds:
                raise ValueError(
                    f"{reference_path} has {len(ref_values)} seeds for "
                    f"{policy}.{raw_key}, need {n_seeds}"
                )
            ours = raw_by_policy[policy][raw_key]
            for seed in range(n_seeds):
                a = round(float(ours[seed]), ndigits)
                b = round(float(ref_values[seed]), ndigits)
                if a != b:
                    policy_mismatches += 1
                    if len(mismatches) < 10:
                        mismatches.append({
                            "policy": policy, "field": raw_key, "seed": seed,
                            "observed": ours[seed], "reference": ref_values[seed],
                        })
        per_policy[policy] = {
            "match": policy_mismatches == 0,
            "mismatched_values": policy_mismatches,
            "compared_values": n_seeds * len(RAW_FIELDS),
        }
    return {
        "reference": reference_label,
        "seeds": list(range(n_seeds)),
        "policies": list(policies),
        "fields": [f[0] for f in RAW_FIELDS],
        "per_policy": per_policy,
        "mismatches": mismatches,
        "mismatch_count": sum(v["mismatched_values"] for v in per_policy.values()),
        "passed": all(v["match"] for v in per_policy.values()),
    }


# --- ordering stability -----------------------------------------------------

def ordering_checks(miss_by_mix):
    """Derive the ordering-stability flag from per-mix mean miss rates.

    ``miss_by_mix`` is {mix: {policy: mean deadline_miss_pct}}. The flag is
    true only when, in every mix, (a) RH and CADR are the two lowest-miss
    policies and (b) every deadline-aware policy (EDF, SPT+Rescue, RH, CADR)
    has a strictly lower mean miss than FIFO. Ranking is by value with the
    policy name as a deterministic tie-break, so dict insertion order cannot
    affect the result.
    """
    per_mix = {}
    for mix, miss in miss_by_mix.items():
        missing = sorted(set(FOCAL_POLICIES) - set(miss))
        if missing:
            raise KeyError(f"mix {mix!r} is missing policies {missing}")
        ranked = sorted(FOCAL_POLICIES, key=lambda p: (miss[p], p))
        below_fifo = {p: miss[p] < miss["FIFO"] for p in DEADLINE_AWARE}
        per_mix[mix] = {
            "ranked_by_miss": list(ranked),
            "two_lowest_miss": list(ranked[:2]),
            "rh_cadr_two_lowest": set(ranked[:2]) == {"RH", "CADR"},
            "deadline_aware_below_fifo": below_fifo,
        }
    stable = all(
        v["rh_cadr_two_lowest"] and all(v["deadline_aware_below_fifo"].values())
        for v in per_mix.values()
    )
    return {
        "claim": (
            "RH and CADR are the two lowest-miss policies, and every "
            "deadline-aware policy is below FIFO on mean miss, in every mix"
        ),
        "per_mix": per_mix,
        "ordering_stable": stable,
    }


# --- determinism ------------------------------------------------------------

def normalized_payload(output):
    """Deep copy of the output with the run-timing fields removed."""
    payload = json.loads(json.dumps(output))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in VOLATILE_METADATA_KEYS:
            metadata.pop(key, None)
    return payload


def normalized_digest(output):
    """sha256 over the canonical JSON of the timing-independent payload."""
    canonical = json.dumps(
        normalized_payload(output), sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- sweep ------------------------------------------------------------------

def run_fleet_mix_sweep(n_seeds=N_SEEDS, policies=FOCAL_POLICIES,
                        verbose=True, reference_path=STATISTICAL_JSON):
    """Run every mix on every seed, then aggregate, pair and check."""
    if n_seeds < 2:
        raise ValueError("at least two seeds are required for a paired test")
    policies = list(policies)
    seeds = list(range(n_seeds))
    raw = {mix: {policy: [] for policy in policies} for mix in MIXES}

    t0 = time.time()
    equivalence = None
    mixes_out = {}
    for mix, slots in MIXES.items():
        if verbose:
            print(f"\n[{mix}] {[s.value for s in slots]}", flush=True)
        for seed in seeds:
            if verbose:
                print(f"  seed {seed:2d}/{seeds[-1]} ...", end=" ", flush=True)
            t_seed = time.time()
            for policy in policies:
                raw[mix][policy].append(
                    run_one(SCHEDULER_MAP[policy], seed, slots))
            if verbose:
                print(f"{time.time() - t_seed:.1f}s", flush=True)
        mixes_out[mix] = {
            "slots": [s.value for s in slots],
            "slot_counts": {
                s.value: sum(1 for x in slots if x == s)
                for s in dict.fromkeys(slots)
            },
            "policies": {
                policy: aggregate_runs(raw[mix][policy]) for policy in policies
            },
        }
        if mix == ANCHOR_MIX:
            # Fail before spending the rest of the sweep if the anchor does
            # not reproduce the canonical Phase 3 run.
            equivalence = check_anchor_equivalence(
                mixes_out[mix]["policies"], n_seeds, policies, reference_path)
            if not equivalence["passed"]:
                raise RuntimeError(
                    "anchor mismatch against "
                    f"{reference_path}: {equivalence['mismatches']}"
                )
            if verbose:
                print(f"  anchor equivalence: passed={equivalence['passed']}",
                      flush=True)
    elapsed = time.time() - t0
    if verbose:
        print(f"\nSweep done in {elapsed:.1f}s.", flush=True)

    paired_tests = []
    for mix in MIXES:
        if mix == ANCHOR_MIX:
            continue
        for policy in policies:
            for metric, raw_key in (
                ("avg_wait_min", "raw_wait_min"),
                ("deadline_miss_pct", "raw_miss_pct"),
            ):
                paired_tests.append({
                    "mix": mix,
                    "reference_mix": ANCHOR_MIX,
                    "policy": policy,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    "direction": "mix minus all-types (positive means the mix is higher)",
                    "test": _paired_t(
                        mixes_out[mix]["policies"][policy][raw_key],
                        mixes_out[ANCHOR_MIX]["policies"][policy][raw_key],
                    ),
                })

    miss_by_mix = {
        mix: {
            policy: mixes_out[mix]["policies"][policy]["miss_pct"]["mean"]
            for policy in policies
        }
        for mix in MIXES
    }
    checks = ordering_checks(miss_by_mix)

    output = {
        "metadata": {
            "n_seeds": n_seeds,
            "seeds": seeds,
            "n_jobs": N_JOBS,
            "arrival_rate": ARRIVAL_RATE,
            "concurrency": CONCURRENCY,
            "duration_hr": DURATION / 3600,
            "policies": policies,
            "anchor_mix": ANCHOR_MIX,
            "elapsed_sec": round(elapsed, 1),
        },
        "mixes": mixes_out,
        "anchor_equivalence": equivalence,
        "paired_tests": paired_tests,
        "ordering_stable": checks["ordering_stable"],
        "ordering_checks": checks["per_mix"],
        "ordering_claim": checks["claim"],
    }
    output["metadata"]["normalized_sha256"] = normalized_digest(output)
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CGRSP Phase 3 - Fleet-Composition Sensitivity")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--policies", choices=("focal", "all"), default="focal",
        help="focal (default, six policies) or all (ten policies, about 9.5 min)",
    )
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    if not STATISTICAL_JSON.exists():
        raise SystemExit(
            f"reference run missing: {STATISTICAL_JSON} is required for the "
            "anchor equivalence check"
        )

    policies = (list(FOCAL_POLICIES) if args.policies == "focal"
                else list(SCHEDULER_MAP))

    print("=" * 60)
    print("PHASE 3 - FLEET-COMPOSITION SENSITIVITY")
    print(f"  Mixes        : {', '.join(MIXES)}")
    print(f"  Seeds / mix  : {args.seeds}")
    print(f"  Policies     : {', '.join(policies)}")
    print(f"  Jobs / run   : {N_JOBS}")
    print("=" * 60)

    output = run_fleet_mix_sweep(
        n_seeds=args.seeds, policies=policies, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 60)
    print(f"{'Mix':<13} {'Policy':<11} {'Wait (min)':<12} {'Miss %':<9} {'Cost ($)':<9}")
    print("-" * 60)
    for mix, m in output["mixes"].items():
        for policy, s in m["policies"].items():
            print(
                f"{mix:<13} {policy:<11} "
                f"{s['wait_min']['mean']:>9.2f}   "
                f"{s['miss_pct']['mean']:>6.2f}   "
                f"{s['cost_usd']['mean']:>7.4f}"
            )

    eq = output["anchor_equivalence"]
    print(f"\nAnchor equivalence vs {eq['reference']}: "
          f"{'PASS' if eq['passed'] else 'FAIL'} "
          f"({eq['mismatch_count']} mismatched values)")
    print(f"Ordering stable across mixes: {output['ordering_stable']}")
    print(f"Normalized sha256: {output['metadata']['normalized_sha256']}")

    if not eq["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
