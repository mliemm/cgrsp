#!/usr/bin/env python3
"""Export a compact, tracked summary of the RunPod pricing-API snapshots.

The raw archive under data/runpod_pricing/ is large and gitignored. Two thesis
figures need it: the snapshot rug in diurnal_model.pdf and the stock-status
history in availability_history.pdf. This script writes the minimum columns
those figures need, so a fresh clone can regenerate them without the archive.

Usage:
    python3 scripts/export_pricing_summary.py

Outputs:
    data/runpod_pricing_summary/snapshot_hours.csv
    data/runpod_pricing_summary/stock_status.csv
"""

from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "runpod_pricing"
OUT = ROOT / "data" / "runpod_pricing_summary"

# Normalise the RunPod API identifiers to the names used in the thesis figures.
FLEET = {
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA RTX A5000": "RTX A5000",
    "NVIDIA RTX A4500": "RTX A4500",
    "NVIDIA RTX A4000": "RTX A4000",
}


def _load(path: Path) -> dict | None:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def export_snapshot_hours() -> int:
    rows = []
    for fp in sorted(glob.glob(str(SRC / "distributions_*.json"))):
        snap = _load(Path(fp))
        if not snap:
            continue
        generated_at = snap.get("generated_at")
        if generated_at:
            rows.append({"generated_at": generated_at})
    out = OUT / "snapshot_hours.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["generated_at"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_stock_status() -> int:
    rows = []
    for fp in sorted(glob.glob(str(SRC / "snapshot_*.json"))):
        snap = _load(Path(fp))
        if not snap:
            continue
        timestamp = snap.get("timestamp", "")
        if len(timestamp) < 13:
            continue
        try:
            # The scraper labelled snapshot timestamps as UTC while its system
            # clock ran on GMT+7; shift by +1 h to Taiwan time, matching the
            # figure code and Section 5.1.1.
            dt_tw = datetime.fromisoformat(timestamp) + timedelta(hours=1)
        except ValueError:
            continue
        for gpu in snap.get("gpus", []):
            name = FLEET.get(gpu.get("id"))
            if not name:
                continue
            status = (gpu.get("lowestPrice") or {}).get("stockStatus") or "Unavailable"
            rows.append(
                {
                    "timestamp_tw": dt_tw.isoformat(),
                    "gpu": name,
                    "status": status,
                }
            )
    out = OUT / "stock_status.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp_tw", "gpu", "status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    if not SRC.exists():
        raise SystemExit(
            f"Missing {SRC}. Restore the gitignored pricing archive first; "
            "the summary is exported from it, not from the thesis outputs."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    n_hours = export_snapshot_hours()
    n_rows = export_stock_status()
    print(f"Wrote {n_hours} snapshot times to {OUT / 'snapshot_hours.csv'}")
    print(f"Wrote {n_rows} per-GPU stock-status rows to {OUT / 'stock_status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
