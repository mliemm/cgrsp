"""Reproducibility guard for experiment entry points.

The simulation is hash-seed-independent at source (provisioning-delay seeding
uses zlib.crc32, not Python's per-process-randomised hash()). Pinning the
interpreter hash seed as well is belt-and-suspenders: it guarantees byte-for-byte
identical re-runs even if some other set/dict iteration order were ever to leak
into a result. Importing this module re-executes the process once with
PYTHONHASHSEED=0 if it is not already set.
"""
import os
import sys
import time

# Since 2026-08-22 the availability model's time-of-day band is computed
# arithmetically from the 08:00 origin (DECISIONS.md #21), so it no longer
# depends on the host clock. The TZ pin is kept as a belt-and-braces no-op so
# any remaining datetime use behaves as on the canonical Asia/Taipei machine.
if os.environ.get("TZ") != "Asia/Taipei":
    os.environ["TZ"] = "Asia/Taipei"
    if hasattr(time, "tzset"):
        time.tzset()

# Re-exec only for real script/module invocations. A "-c" command or the
# interactive REPL cannot be reconstructed from sys.argv, so skip those (they
# are ad-hoc and not used to generate thesis results). Also skip under pytest:
# execv during test collection replaces the pytest process and swallows all
# output (the suite dies with exit 2 and zero bytes); the simulation is
# hash-seed-independent at source, so tests do not need the pin.
if (
    os.environ.get("PYTHONHASHSEED") != "0"
    and "pytest" not in sys.modules
    and sys.argv
    and sys.argv[0] not in ("-c", "")
):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
