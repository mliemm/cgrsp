"""Exclude rendering jobs that ran while more than five jobs were in flight.

C_max = 5 is RunPod's default per-account Serverless worker quota. The production
log contains bursts above that limit. To keep the calibration data inside the
quota the simulator assumes, every job whose
[start, completion] interval overlaps a moment with more than five jobs in
flight is removed. The completion time is the log timestamp and the start time is
that timestamp minus total_time_s. All environments (PROD, DEV, DEV_2, untagged)
count towards concurrency.

Usage (from the repository root):
    python3 scripts/filter_over_five_concurrency.py

Reads data/unused_after_filter/rendering_performance_data_unfiltered.csv and writes
data/rendering_performance_data.csv (kept jobs) and
data/unused_after_filter/rendering_performance_excluded_over5.csv (removed jobs).
"""
import csv
import datetime as dt
from pathlib import Path

LIMIT = 5
DATA = Path(__file__).resolve().parent.parent / "data"
UNUSED = DATA / "unused_after_filter"
SRC = UNUSED / "rendering_performance_data_unfiltered.csv"
KEPT = DATA / "rendering_performance_data.csv"
EXCLUDED = UNUSED / "rendering_performance_excluded_over5.csv"


def interval(row):
    end = dt.datetime.strptime(row["timestamp"], "%Y:%m:%dT%H:%M:%S")
    return end - dt.timedelta(seconds=float(row["total_time_s"])), end


def over_limit_jobs(rows):
    """Indices of jobs in flight at any moment when more than LIMIT are in flight."""
    events = []
    for index, row in enumerate(rows):
        start, end = interval(row)
        events.append((start, 1, index))
        events.append((end, -1, index))
    # At equal times a completion is processed before a start, so touching intervals do not overlap.
    events.sort(key=lambda event: (event[0], event[1]))
    active, excluded = set(), set()
    for _, step, index in events:
        if step == 1:
            active.add(index)
            if len(active) > LIMIT:
                excluded |= active
        else:
            active.discard(index)
    return excluded


def peak(rows):
    events = []
    for row in rows:
        start, end = interval(row)
        events += [(start, 1), (end, -1)]
    events.sort(key=lambda event: (event[0], event[1]))
    level = top = 0
    for _, step in events:
        level += step
        top = max(top, level)
    return top


def main():
    with SRC.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    removed = over_limit_jobs(rows)
    kept_rows = [row for index, row in enumerate(rows) if index not in removed]
    removed_rows = [row for index, row in enumerate(rows) if index in removed]
    assert peak(kept_rows) <= LIMIT, "filtered data still exceeds the limit"
    for path, subset in ((KEPT, kept_rows), (EXCLUDED, removed_rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(subset)
    print(f"input {len(rows)}, kept {len(kept_rows)}, excluded {len(removed_rows)}, "
          f"peak before {peak(rows)}, after {peak(kept_rows)}")


if __name__ == "__main__":
    main()
