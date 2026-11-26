#!/usr/bin/env bash
# Verify a fresh clone of the CGRSP thesis repository.
#
# Tiers:
#   --fast     venv install, pytest, validate_numbers.py, normalized JSON hash
#              compare and a hash-compare self-test. Budget: 5 min.
#   --rebuild  --fast plus generate_data.py and scripts/build_thesis.sh, with a
#              page-count comparison. Budget: 15 min.
#   --full     --rebuild plus a determinism spot-check on two cheap experiment
#              entry points, comparing normalized JSON before and after.
#              Budget: 30 min. This is NOT run_all.sh; see below.
#
# The script clones HEAD into mktemp -d and never writes inside the source
# repository. It exits 1 on any failed check, 2 when a tier was skipped for a
# missing prerequisite (a skipped tier is not a pass), and 0 when every
# executed check passed.
#
# docs/release_sha256.txt is produced later. If it is absent, the normalized
# hashes are computed and printed with a PLACEHOLDER marker; the compare is not
# claimed as a pass. Pass --manifest PATH to use a manifest elsewhere.
#
# run_all.sh (every experiment, about an hour or more) is outside the --full
# budget. --full reruns CGRSP/main.py and CGRSP/phase3_cr_sensitivity.py and
# compares every output file it can see; it does not rerun the whole suite.
set -uo pipefail

usage() {
  cat <<'TEXT'
Usage: scripts/verify_fresh_clone.sh --fast|--rebuild|--full [options]

Options:
  --manifest PATH   compare CGRSP/Outputs JSON against this manifest instead of
                    docs/release_sha256.txt
  --source PATH     repository to clone (default: the repository containing
                    this script)
  --keep            keep the temporary clone and print its path
  -h, --help        show this help

Exit codes: 0 all executed checks passed; 1 a check failed; 2 a tier or
sub-tier was skipped because a prerequisite was missing (not a pass).
TEXT
}

TIER=""
MANIFEST=""
KEEP=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3.12}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fast|--rebuild|--full)
      if [ -n "$TIER" ]; then
        echo "error: choose exactly one tier" >&2
        exit 64
      fi
      TIER="${1#--}"
      ;;
    --manifest)
      shift
      [ "$#" -gt 0 ] || { echo "error: --manifest needs a path" >&2; exit 64; }
      MANIFEST="$1"
      ;;
    --source)
      shift
      [ "$#" -gt 0 ] || { echo "error: --source needs a path" >&2; exit 64; }
      SOURCE="$1"
      ;;
    --keep)
      KEEP=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
  shift
done

if [ -z "$TIER" ]; then
  echo "error: choose --fast, --rebuild or --full" >&2
  usage >&2
  exit 64
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
INCOMPLETE=0

section() { printf '\n== %s ==\n' "$*"; }
pass()    { printf 'PASS  %s\n' "$*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail()    { printf 'FAIL  %s\n' "$*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
skip()    { printf 'SKIP  %s\n' "$*"; SKIP_COUNT=$((SKIP_COUNT + 1)); }

for command_name in git mktemp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing prerequisite: $command_name" >&2
    exit 2
  fi
done
if command -v shasum >/dev/null 2>&1; then
  hash_text() { shasum -a 256; }
elif command -v sha256sum >/dev/null 2>&1; then
  hash_text() { sha256sum; }
else
  echo "missing prerequisite: shasum or sha256sum" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "missing prerequisite: $PYTHON_BIN (set PYTHON=... to override)" >&2
  exit 2
fi
if ! git -C "$SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: source is not a git repository: $SOURCE" >&2
  exit 2
fi

source_fingerprint() {
  {
    git -C "$SOURCE" status --porcelain=v1
    git -C "$SOURCE" diff --no-color
  } | hash_text 2>/dev/null | awk '{print $1}'
}
SOURCE_STATE_BEFORE="$(source_fingerprint)"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cgrsp-verify.XXXXXX")" || {
  echo "missing prerequisite: mktemp could not create a directory" >&2
  exit 2
}
cleanup() {
  if [ "$KEEP" -eq 1 ]; then
    printf '\nkept: %s\n' "$TMP_ROOT"
  else
    rm -rf "$TMP_ROOT"
  fi
}
trap cleanup EXIT

CLONE="$TMP_ROOT/clone"
VENV="$TMP_ROOT/venv"
PY="$VENV/bin/python"
STARTED_AT=$SECONDS

printf 'CGRSP fresh-clone verification\n'
printf 'tier   : %s\n' "$TIER"
printf 'source : %s\n' "$SOURCE"
printf 'clone  : %s\n' "$CLONE"
printf 'python : %s\n' "$PYTHON_BIN"

section "clone"
if ! git clone --quiet --no-hardlinks -- "$SOURCE" "$CLONE" >"$TMP_ROOT/clone.log" 2>&1; then
  fail "git clone $SOURCE $CLONE"
  cat "$TMP_ROOT/clone.log"
  exit 1
fi
HEAD_SHA="$(git -C "$CLONE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
pass "cloned HEAD $HEAD_SHA into a temporary directory"
if [ ! -e "$CLONE/data/runpod_pricing" ] && [ ! -e "$CLONE/data/rendering_jobs" ]; then
  pass "clone omits data/runpod_pricing and data/rendering_jobs"
else
  printf 'NOTE  the clone contains a data directory that the tracked tree should not have; the missing-data condition was not exercised.\n'
fi
if [ -n "$(git -C "$SOURCE" status --porcelain=v1)" ]; then
  printf 'NOTE  the source worktree has uncommitted or untracked changes; the clone is HEAD %s only.\n' "$HEAD_SHA"
fi

section "Python environment"
VENV_START=$SECONDS
if ! "$PYTHON_BIN" -m venv "$VENV" >"$TMP_ROOT/venv.log" 2>&1; then
  fail "create venv with $PYTHON_BIN"
  tail -30 "$TMP_ROOT/venv.log"
  exit 1
fi
if ! "$PY" -m pip install --quiet --disable-pip-version-check \
    -r "$CLONE/CGRSP/requirements.txt" >>"$TMP_ROOT/venv.log" 2>&1; then
  fail "pip install -r CGRSP/requirements.txt (log: $TMP_ROOT/venv.log)"
  tail -30 "$TMP_ROOT/venv.log"
  exit 1
fi
"$PY" --version | sed 's/^/      /'
pass "venv installed in $((SECONDS - VENV_START)) s"

section "_repro re-exec with PYTHONHASHSEED unset"
cat >"$TMP_ROOT/repro_probe.py" <<'PY'
import os
import sys

print("before_import=%s" % os.environ.get("PYTHONHASHSEED"), flush=True)
sys.path.insert(0, sys.argv[1])
import _repro  # noqa: F401
print("after_import=%s" % os.environ.get("PYTHONHASHSEED"), flush=True)
PY
if REPRO_OUT="$(env -u PYTHONHASHSEED "$PY" "$TMP_ROOT/repro_probe.py" \
    "$CLONE/CGRSP" 2>&1)"; then
  printf '%s\n' "$REPRO_OUT" | sed 's/^/      /'
  FIRST_LINE="$(printf '%s\n' "$REPRO_OUT" | sed -n '1p')"
  SECOND_LINE="$(printf '%s\n' "$REPRO_OUT" | sed -n '2p')"
  THIRD_LINE="$(printf '%s\n' "$REPRO_OUT" | sed -n '3p')"
  if [ "$FIRST_LINE" = "before_import=None" ] \
      && [ "$SECOND_LINE" = "before_import=0" ] \
      && [ "$THIRD_LINE" = "after_import=0" ]; then
    pass "PYTHONHASHSEED unset, _repro re-executed with 0"
  else
    fail "_repro re-exec output did not show the unset -> 0 transition"
  fi
else
  fail "_repro probe failed"
  printf '%s\n' "$REPRO_OUT" | sed 's/^/      /'
fi

section "pytest"
if ( cd "$CLONE" && env -u PYTHONHASHSEED "$PY" -m pytest -q ) \
    >"$TMP_ROOT/pytest.log" 2>&1; then
  grep -E "passed|failed|error" "$TMP_ROOT/pytest.log" | tail -1 | sed 's/^/      /'
  pass "pytest"
else
  fail "pytest (log: $TMP_ROOT/pytest.log)"
  tail -40 "$TMP_ROOT/pytest.log"
fi

section "validate_numbers.py"
if ( cd "$CLONE" && env -u PYTHONHASHSEED "$PY" validate_numbers.py ) \
    >"$TMP_ROOT/validate.log" 2>&1; then
  grep "validate_numbers:" "$TMP_ROOT/validate.log" | tail -1 | sed 's/^/      /'
  pass "validate_numbers.py"
else
  fail "validate_numbers.py (log: $TMP_ROOT/validate.log)"
  tail -40 "$TMP_ROOT/validate.log"
fi

# ---------------------------------------------------------------------------
# Normalized JSON hashes.
#
# Timing fields are removed before hashing, per REPRODUCE.md section 7:
# timing keys (elapsed_sec, train_elapsed_sec, eval_elapsed_sec, runtime_s,
# generated_at) wherever they appear.
# Section 7 names beta_sensitivity.json, pressure_sensitivity.json and
# prior_robustness.json for runtime_s; diagnostics.json also carries it at the
# top level, so the helper removes runtime_s wherever it is a top-level key
# rather than letting a rerun fail on a timing field.
# ---------------------------------------------------------------------------
cat >"$TMP_ROOT/normhash.py" <<'PY'
#!/usr/bin/env python3
"""Normalized SHA-256 comparison for CGRSP/Outputs JSON."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

TIMING_KEYS = {"elapsed_sec", "train_elapsed_sec", "eval_elapsed_sec", "runtime_s", "generated_at"}
OUTPUTS = Path("CGRSP") / "Outputs"


def normalize(document):
    """Return (document without timing fields, sorted removed field names)."""
    removed = set()

    def walk(node):
        if isinstance(node, dict):
            result = {}
            for key, value in node.items():
                if key in TIMING_KEYS:
                    removed.add(key)
                    continue
                result[key] = walk(value)
            return result
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(document), sorted(removed)


def canonical(document):
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def json_files(root):
    """Tracked CGRSP/Outputs JSON, so ignored derived files never enter a manifest.

    CGRSP/.gitignore ignores Outputs/, while 21 release JSON files are tracked
    with force-add. A worktree can therefore hold additional ignored JSON that
    a fresh clone does not have. Use git to enumerate the release set; fall
    back to a glob when root is not a repository (for example the self-test's
    temporary root).
    """
    outputs = root / OUTPUTS
    if (root / ".git").exists():
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--", "CGRSP/Outputs"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            completed = None
        if completed is not None:
            paths = []
            for raw in completed.stdout.split(b"\0"):
                if not raw.endswith(b".json"):
                    continue
                path = root / raw.decode("utf-8")
                if path.is_file():
                    paths.append(path)
            if paths:
                return sorted(paths)
    return sorted(path for path in outputs.rglob("*.json") if path.is_file())


def digest(root, path):
    document = json.loads(path.read_text(encoding="utf-8"))
    normalized, removed = normalize(document)
    return hashlib.sha256(canonical(normalized)).hexdigest(), removed


def relpath(root, path):
    return path.relative_to(root).as_posix()


def read_manifest(path):
    expected = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{path}:{line_number}: expected '<sha256>  <path>'")
        expected[parts[1].strip().lstrip("*")] = parts[0].lower()
    return expected


def write_manifest(root, out):
    lines = [
        "# SHA-256 of canonical JSON in CGRSP/Outputs.",
        "# Timing fields removed before hashing: metadata.elapsed_sec,",
        "# metadata.generated_at, and top-level runtime_s.",
        "# Format: <sha256>  <repo-relative path>",
    ]
    for path in json_files(root):
        value, _ = digest(root, path)
        lines.append(f"{value}  {relpath(root, path)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_manifest(root, manifest_path, quiet=False):
    try:
        expected = read_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        print(f"FAIL  manifest: {exc}")
        return 1
    actual = {}
    for path in json_files(root):
        actual[relpath(root, path)] = digest(root, path)

    failures = 0
    for rel in sorted(actual):
        value, removed = actual[rel]
        if rel not in expected:
            print(f"FAIL  {rel}: not present in the manifest")
            failures += 1
        elif expected[rel] != value:
            fields = ", ".join(removed) if removed else "none"
            print(f"FAIL  {rel}: normalized SHA-256 mismatch")
            print(f"      normalized fields excluded: {fields}")
            print(f"      expected {expected[rel]}")
            print(f"      actual   {value}")
            failures += 1
        elif not quiet:
            print(f"ok    {rel}")
    for rel in sorted(set(expected) - set(actual)):
        print(f"FAIL  {rel}: in the manifest but absent from the clone")
        failures += 1
    if not failures:
        print(f"ok    {len(actual)} normalized JSON files match {manifest_path}")
    return 1 if failures else 0


def snapshot(root, out):
    data = {}
    for path in json_files(root):
        document = json.loads(path.read_text(encoding="utf-8"))
        data[relpath(root, path)] = normalize(document)[0]
    out.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def diff_values(before, after, trail, out, limit=25):
    if len(out) >= limit:
        return
    if type(before) is not type(after):
        out.append(f"{trail}: type {type(before).__name__} != {type(after).__name__}")
        return
    if isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                out.append(f"{trail}.{key}: missing before")
            elif key not in after:
                out.append(f"{trail}.{key}: missing after")
            else:
                diff_values(before[key], after[key], f"{trail}.{key}", out, limit)
            if len(out) >= limit:
                return
    elif isinstance(before, list):
        if len(before) != len(after):
            out.append(f"{trail}: length {len(before)} != {len(after)}")
        for index, (left, right) in enumerate(zip(before, after)):
            diff_values(left, right, f"{trail}[{index}]", out, limit)
            if len(out) >= limit:
                return
    elif before != after:
        out.append(f"{trail}: {before!r} != {after!r}")


def diff_snapshots(before_path, after_path):
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    failures = 0
    for rel in sorted(set(before) | set(after)):
        if rel not in before:
            print(f"FAIL  {rel}: file appeared after the rerun")
            failures += 1
            continue
        if rel not in after:
            print(f"FAIL  {rel}: file disappeared after the rerun")
            failures += 1
            continue
        differences = []
        diff_values(before[rel], after[rel], rel, differences)
        if differences:
            failures += 1
            for line in differences:
                print(f"FAIL  {line}")
    if not failures:
        print(f"ok    {len(before)} normalized JSON files unchanged")
    return 1 if failures else 0


def selftest(root, work):
    """Prove that the compare fails on a result mutation and ignores timing."""
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    (work / OUTPUTS).mkdir(parents=True)

    with_timing = None
    with_metadata = None
    for path in json_files(root):
        document = json.loads(path.read_text(encoding="utf-8"))
        _, removed = normalize(document)
        if "runtime_s" in removed and with_timing is None:
            with_timing = path
        if "elapsed_sec" in removed and with_metadata is None:
            with_metadata = path
        if with_timing and with_metadata:
            break

    if with_timing is None or with_metadata is None:
        print("SKIP  hash self-test: no file with runtime_s and no file with elapsed_sec")
        return 0

    def materialize(path):
        target = work / relpath(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return target

    def run_check():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = check_manifest(work, work / "manifest.txt")
        return code, buffer.getvalue()

    timing_target = materialize(with_timing)
    original_timing = json.loads(timing_target.read_text(encoding="utf-8"))
    metadata_target = materialize(with_metadata)
    original_metadata = json.loads(metadata_target.read_text(encoding="utf-8"))
    write_manifest(work, work / "manifest.txt")

    code, _ = run_check()
    if code != 0:
        print("FAIL  hash self-test: baseline did not match")
        return 1

    mutated = json.loads(json.dumps(original_timing))
    changed = False
    for key in ("description", "note"):
        if isinstance(mutated.get(key), str):
            mutated[key] = mutated[key] + " MUTATED"
            changed = True
            break
    if not changed:
        mutated["_mutated"] = "MUTATED"
    timing_target.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    code, _ = run_check()
    if code == 0:
        print("FAIL  hash self-test: a result mutation was not detected")
        return 1
    print("PASS  hash self-test: a result mutation is detected")

    mutated = json.loads(json.dumps(original_timing))
    mutated["runtime_s"] = float(mutated.get("runtime_s", 0)) + 1234.5
    timing_target.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    code, _ = run_check()
    if code != 0:
        print("FAIL  hash self-test: a top-level runtime_s mutation was not ignored")
        return 1
    print("PASS  hash self-test: a top-level runtime_s mutation is ignored")

    mutated = json.loads(json.dumps(original_metadata))
    mutated.setdefault("metadata", {})["elapsed_sec"] = 999999.0
    metadata_target.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    code, _ = run_check()
    if code != 0:
        print("FAIL  hash self-test: a metadata.elapsed_sec mutation was not ignored")
        return 1
    print("PASS  hash self-test: a metadata.elapsed_sec mutation is ignored")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    write_parser = sub.add_parser("write")
    write_parser.add_argument("--root", required=True)
    write_parser.add_argument("--out", required=True)

    check_parser = sub.add_parser("check")
    check_parser.add_argument("--root", required=True)
    check_parser.add_argument("--manifest", required=True)
    check_parser.add_argument("--quiet", action="store_true")

    snapshot_parser = sub.add_parser("snapshot")
    snapshot_parser.add_argument("--root", required=True)
    snapshot_parser.add_argument("--out", required=True)

    diff_parser = sub.add_parser("diff")
    diff_parser.add_argument("--before", required=True)
    diff_parser.add_argument("--after", required=True)

    selftest_parser = sub.add_parser("selftest")
    selftest_parser.add_argument("--root", required=True)
    selftest_parser.add_argument("--work", required=True)

    args = parser.parse_args(argv)
    if args.command == "write":
        write_manifest(Path(args.root), Path(args.out))
        print(f"ok    wrote {args.out}")
        return 0
    if args.command == "check":
        return check_manifest(Path(args.root), Path(args.manifest), args.quiet)
    if args.command == "snapshot":
        snapshot(Path(args.root), Path(args.out))
        print(f"ok    wrote snapshot {args.out}")
        return 0
    if args.command == "diff":
        return diff_snapshots(args.before, args.after)
    if args.command == "selftest":
        return selftest(Path(args.root), Path(args.work))
    parser.error("unknown command")
    return 64


if __name__ == "__main__":
    sys.exit(main())
PY

section "normalized JSON hash compare self-test"
if "$PY" "$TMP_ROOT/normhash.py" selftest --root "$CLONE" \
    --work "$TMP_ROOT/selftest" >"$TMP_ROOT/selftest.log" 2>&1; then
  sed 's/^/      /' "$TMP_ROOT/selftest.log"
  pass "hash self-test"
else
  fail "hash self-test (log: $TMP_ROOT/selftest.log)"
  sed 's/^/      /' "$TMP_ROOT/selftest.log"
fi

section "normalized JSON hash compare"
MANIFEST_MISSING=0
if [ -n "$MANIFEST" ]; then
  if [ ! -f "$MANIFEST" ]; then
    fail "manifest not found: $MANIFEST"
    MANIFEST_MISSING=1
  fi
elif [ -f "$CLONE/docs/release_sha256.txt" ]; then
  MANIFEST="$CLONE/docs/release_sha256.txt"
elif [ -f "$SOURCE/docs/release_sha256.txt" ]; then
  MANIFEST="$SOURCE/docs/release_sha256.txt"
  printf 'note  using the source worktree manifest: %s\n' "$MANIFEST"
fi

if [ "$MANIFEST_MISSING" -eq 1 ]; then
  : # the missing manifest was already reported as a failed check
elif [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ]; then
  if "$PY" "$TMP_ROOT/normhash.py" check --root "$CLONE" \
      --manifest "$MANIFEST" --quiet 2>&1 | tee "$TMP_ROOT/hash_check.log"; then
    pass "normalized JSON hashes match $MANIFEST"
  else
    fail "normalized JSON hash mismatch (see the FAIL lines above; log: $TMP_ROOT/hash_check.log)"
  fi
else
  "$PY" "$TMP_ROOT/normhash.py" write --root "$CLONE" \
    --out "$TMP_ROOT/normalized_manifest.txt" >/dev/null 2>&1
  HASH_COUNT="$(grep -v '^#' "$TMP_ROOT/normalized_manifest.txt" | wc -l | tr -d ' ')"
  printf 'PLACEHOLDER: docs/release_sha256.txt is absent; computed %s normalized file hashes.\n' "$HASH_COUNT"
  printf '             written to %s (not inside the repository).\n' "$TMP_ROOT/normalized_manifest.txt"
  printf '             This is not a pass of the hash compare.\n'
  skip "normalized JSON hash compare (manifest placeholder; --manifest PATH to compare)"
fi

if [ "$TIER" = "fast" ]; then
  section "source worktree"
  SOURCE_STATE_AFTER="$(source_fingerprint)"
  if [ "$SOURCE_STATE_BEFORE" = "$SOURCE_STATE_AFTER" ]; then
    pass "the source worktree was not modified"
  else
    fail "the source worktree changed during the run"
  fi

  section "summary"
  printf 'checks passed : %s\n' "$PASS_COUNT"
  printf 'checks failed : %s\n' "$FAIL_COUNT"
  printf 'checks skipped: %s\n' "$SKIP_COUNT"
  printf 'elapsed       : %s s\n' "$((SECONDS - STARTED_AT))"
  if [ "$((SECONDS - STARTED_AT))" -gt 300 ]; then
    printf 'WARN  --fast exceeded its 5 min budget on this machine\n'
  fi
  if [ "$FAIL_COUNT" -gt 0 ]; then
    printf 'RESULT: FAIL\n'
    exit 1
  fi
  printf 'RESULT: PASS'
  if [ "$SKIP_COUNT" -gt 0 ]; then
    printf ' (with %s skipped check(s); a skipped check is not a pass)' "$SKIP_COUNT"
  fi
  printf '\n'
  exit 0
fi

# ---------------------------------------------------------------------------
# Rebuild tier.
# ---------------------------------------------------------------------------
section "rebuild: generate_data.py"
BUILD_START=$SECONDS
if [ -n "$(git -C "$CLONE" status --porcelain=v1)" ]; then
  fail "the clone is dirty before generate_data.py"
  git -C "$CLONE" status --porcelain=v1
fi
if ( cd "$CLONE" && "$PY" generate_data.py ) >"$TMP_ROOT/generate.log" 2>&1; then
  CHANGED="$(git -C "$CLONE" status --porcelain=v1)"
  if [ -z "$CHANGED" ]; then
    pass "generate_data.py reproduced thesis/generated byte for byte"
  else
    fail "generate_data.py changed tracked files:"
    printf '%s\n' "$CHANGED" | sed 's/^/      /'
  fi
else
  fail "generate_data.py (log: $TMP_ROOT/generate.log)"
  tail -40 "$TMP_ROOT/generate.log"
fi

section "rebuild: thesis PDF"
if ! command -v xelatex >/dev/null 2>&1 || ! command -v bibtex >/dev/null 2>&1; then
  skip "TeX rebuild (xelatex or bibtex not found); a skipped tier is not a pass"
  INCOMPLETE=1
else
  page_count() {
    if command -v pdfinfo >/dev/null 2>&1; then
      pdfinfo "$1" 2>/dev/null | awk '/^Pages:/ {print $2; exit}'
    elif command -v mdls >/dev/null 2>&1; then
      mdls -name kMDItemNumberOfPages -raw "$1" 2>/dev/null \
        | tr -cd '0-9'
    fi
  }
  PAGES_BEFORE="$(page_count "$CLONE/thesis/thesis.pdf")"
  if ! ( cd "$CLONE" && ./scripts/build_thesis.sh ) >"$TMP_ROOT/build.log" 2>&1; then
    fail "scripts/build_thesis.sh (log: $TMP_ROOT/build.log)"
    tail -40 "$TMP_ROOT/build.log"
  else
    PAGES_AFTER="$(page_count "$CLONE/thesis/thesis.pdf")"
    printf '      pages before: %s\n' "${PAGES_BEFORE:-unknown}"
    printf '      pages after : %s\n' "${PAGES_AFTER:-unknown}"
    if [ -n "$PAGES_BEFORE" ] && [ -n "$PAGES_AFTER" ]; then
      if [ "$PAGES_BEFORE" = "$PAGES_AFTER" ]; then
        pass "thesis rebuild page count unchanged ($PAGES_AFTER)"
      else
        fail "thesis rebuild page count changed: $PAGES_BEFORE -> $PAGES_AFTER"
      fi
    else
      skip "page count comparison (pdfinfo and mdls unavailable)"
    fi
    if [ -f "$CLONE/public/thesis.pdf" ] \
        && cmp -s "$CLONE/thesis/thesis.pdf" "$CLONE/public/thesis.pdf"; then
      pass "public/thesis.pdf matches thesis/thesis.pdf"
    else
      fail "public/thesis.pdf missing or different from thesis/thesis.pdf"
    fi
    FINAL_TEX_LOG="$CLONE/thesis/thesis-build-pass4.log"
    if [ -f "$FINAL_TEX_LOG" ] \
        && grep -E "Undefined control sequence|There were undefined references|LaTeX Warning: Reference .* undefined|Package natbib Warning: Citation .* undefined" \
        "$FINAL_TEX_LOG" >"$TMP_ROOT/tex_warnings.log" 2>/dev/null; then
      fail "the final TeX pass contains undefined references or control sequences:"
      head -10 "$TMP_ROOT/tex_warnings.log" | sed 's/^/      /'
    else
      pass "the final TeX pass has no undefined references or control sequences"
    fi
  fi
fi
printf '      rebuild elapsed: %s s\n' "$((SECONDS - BUILD_START))"
if [ "$((SECONDS - BUILD_START))" -gt 900 ]; then
  printf 'WARN  rebuild exceeded its 15 min budget on this machine\n'
fi

# ---------------------------------------------------------------------------
# Full tier: representative determinism spot-check.
# ---------------------------------------------------------------------------
if [ "$TIER" = "full" ]; then
  section "full: determinism spot-check"
  if ! "$PY" "$TMP_ROOT/normhash.py" snapshot --root "$CLONE" \
      --out "$TMP_ROOT/pre_run.json" >/dev/null 2>&1; then
    fail "could not snapshot normalized JSON before the reruns"
  else
    RERUN_START=$SECONDS
    if ( cd "$CLONE" && env -u PYTHONHASHSEED PYTHONPATH="$CLONE/CGRSP" \
        "$PY" CGRSP/main.py --output CGRSP/Outputs/runpod_scheduler ) \
        >"$TMP_ROOT/rerun_main.log" 2>&1; then
      pass "CGRSP/main.py reran"
    else
      fail "CGRSP/main.py (log: $TMP_ROOT/rerun_main.log)"
      tail -30 "$TMP_ROOT/rerun_main.log"
    fi
    if ( cd "$CLONE" && env -u PYTHONHASHSEED PYTHONPATH="$CLONE/CGRSP" \
        "$PY" CGRSP/phase3_cr_sensitivity.py ) \
        >"$TMP_ROOT/rerun_cr.log" 2>&1; then
      pass "CGRSP/phase3_cr_sensitivity.py reran"
    else
      fail "CGRSP/phase3_cr_sensitivity.py (log: $TMP_ROOT/rerun_cr.log)"
      tail -30 "$TMP_ROOT/rerun_cr.log"
    fi
    printf '      rerun elapsed: %s s\n' "$((SECONDS - RERUN_START))"

    if "$PY" "$TMP_ROOT/normhash.py" snapshot --root "$CLONE" \
        --out "$TMP_ROOT/post_run.json" >/dev/null 2>&1; then
      if "$PY" "$TMP_ROOT/normhash.py" diff \
          --before "$TMP_ROOT/pre_run.json" --after "$TMP_ROOT/post_run.json" \
          2>&1 | tee "$TMP_ROOT/diff.log"; then
        pass "normalized JSON unchanged after the representative reruns"
      else
        fail "a result field changed after the representative reruns (see the FAIL lines above)"
      fi
    else
      fail "could not snapshot normalized JSON after the reruns"
    fi

    if [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ]; then
      if "$PY" "$TMP_ROOT/normhash.py" check --root "$CLONE" \
          --manifest "$MANIFEST" --quiet 2>&1 | tee "$TMP_ROOT/hash_after.log"; then
        pass "normalized JSON still matches the manifest after the reruns"
      else
        fail "normalized JSON no longer matches the manifest after the reruns (see the FAIL lines above)"
      fi
    fi
  fi
  printf '      full elapsed: %s s\n' "$((SECONDS - BUILD_START))"
  if [ "$((SECONDS - BUILD_START))" -gt 1800 ]; then
    printf 'WARN  --full exceeded its 30 min budget on this machine\n'
  fi
fi

section "source worktree"
SOURCE_STATE_AFTER="$(source_fingerprint)"
if [ "$SOURCE_STATE_BEFORE" = "$SOURCE_STATE_AFTER" ]; then
  pass "the source worktree was not modified"
else
  fail "the source worktree changed during the run"
fi

section "summary"
printf 'checks passed : %s\n' "$PASS_COUNT"
printf 'checks failed : %s\n' "$FAIL_COUNT"
printf 'checks skipped: %s\n' "$SKIP_COUNT"
printf 'elapsed       : %s s\n' "$((SECONDS - STARTED_AT))"
if [ "$FAIL_COUNT" -gt 0 ]; then
  printf 'RESULT: FAIL\n'
  exit 1
fi
if [ "$INCOMPLETE" -gt 0 ]; then
  printf 'RESULT: INCOMPLETE (a skipped tier is not a pass)\n'
  exit 2
fi
printf 'RESULT: PASS (reproducibility checks only; correctness and sign-off are human)\n'
exit 0
