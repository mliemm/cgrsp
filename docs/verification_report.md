# Verification Report

This report records the automated reproduction runs for the journal package. It is not a human verification sign-off.

## Commits

| Run | Commit | Scope |
|---|---|---|
| Fast tier, Windows/WSL | `738b119` | Clone, line endings, 207 tests, `validate_numbers.py`, 23-file hash compare, `scripts/verify_fresh_clone.sh --fast` |
| Fast tier, local macOS | `738b119` | Same fast tier on the reference platform |
| Fast tier, Linux (Ubuntu 22.04) | `7883488` | Same fast tier on Python 3.12.16; 82 s |
| Full pipeline, local macOS | `8ba4756` | `run_all.sh`, complete; 187 pages |

The commits after `738b119` change documentation, the manuscript, the Drive sync script, the verification harness, and the thesis PDF bytes. `CGRSP/` and `data/` are unchanged, so the Linux run at `7883488` exercises the same simulation code as the full run at `8ba4756`.

## Results

| Check | Result | Evidence |
|---|---|---|
| Clone | PASS | Windows checkout and WSL clone at `738b119`; Linux clone at `7883488`; clean worktree |
| Line endings | PASS | `git ls-files --eol` reports `i/lf w/lf attr/text eol=lf` for `.sh` and `.py`; zero CR bytes |
| Test suite | PASS | 207 passed on WSL 3.12.3, native Windows 3.13.15, Linux 3.12.16, and macOS 3.12 |
| `validate_numbers.py` | PASS | Native Windows with `PYTHONIOENCODING=cp1252`, no `PYTHONUTF8`; Linux; macOS |
| Normalized hash compare | PASS | 23 of 23 JSON files match `docs/release_sha256.txt`; mutation self-test passes |
| Fresh-clone harness | PASS | 9 checks passed, 0 failed, 0 skipped; 65 s on Windows/WSL and macOS, 82 s on Linux |
| Full `run_all.sh` | PASS | Exit 0; generated tables byte-identical; 23 of 23 normalized JSON; 187-page PDF text identical; sampled figures pixel-identical |

## Full pipeline result

The full run was executed on a clean clone at `8ba4756`:

1. `./run_all.sh` exited 0. It ran Phase 2, every Phase 3 sweep, the baseline experiment, the fleet-mix sweep, Phase 4, table and figure generation, validation, and the thesis build.
2. `thesis/generated/*.tex` has no diff against the committed fragments. Every table and macro fragment is byte-identical.
3. The normalized JSON compare matches 23 of 23 files after timing fields are removed.
4. `thesis/thesis.pdf` is 187 pages. A `pdftotext` comparison against the committed PDF is empty.
5. The regenerated figure PDFs differ in binary metadata but not in size. Four sampled figures, `ttest_effects`, `phase4_monthly`, `availability_history` and `gpu_heatmap`, render pixel-identical. The remaining figures were not pixel-compared.
6. The full run exposed one normalizer gap: nested per-episode `elapsed_sec` values and `metadata.train_elapsed_sec`/`eval_elapsed_sec` were not stripped, so `baselines.json` first failed the hash compare. The normalizer now removes all timing keys wherever they appear, and the manifest was regenerated. Both the committed tree and the full-run clone then matched 23 of 23.

## Fixes made from the verification

1. `.gitattributes` forces LF for `.sh` and `.py`, so a Windows checkout with `core.autocrlf=true` no longer breaks WSL bash.
2. `validate_numbers.py` forces UTF-8 output and explicit UTF-8 reads, so it no longer crashes on the Chinese abstract or the box-drawing output under a legacy Windows code page.
3. All shell entry points are mode `100755`.
4. `REPRODUCE.md` and `docs/verification_checklist.md` document the Windows/WSL requirement and the `git checkout-index -a -f` repair for an older checkout.
5. The normalized-hash helper strips `elapsed_sec`, `train_elapsed_sec`, `eval_elapsed_sec`, `runtime_s` and `generated_at` wherever they appear, so a full rerun does not fail on timing fields.

## Residual gaps

1. A human sign-off is still required. These runs are automated and do not replace it.
2. The Windows and Linux machines have no TeX installation, so the PDF rebuild was not checked there. The local macOS build is 187 pages with zero undefined references and zero overfull boxes, and the full run rebuilt it.
3. The fast tier is the 5-minute check for a human verifier. The full `run_all.sh` result above is the reference for numeric reproduction.
4. The reference environment is macOS with Python 3.14.0. The Windows/WSL run used Python 3.12.3 and 3.13.15, and the Linux run used Python 3.12.16, all with the pinned dependencies. Ubuntu 22.04 ships Python 3.10, so the Linux check used a user-level Python 3.12.16 from `uv`. These runs are portability checks, not the original environment.
5. Four figures were sampled for pixel identity, not all of them. The table fragments, JSON, and PDF text were compared exhaustively.

## Human sign-off

The automated checks above are reproducible. A human verifier should confirm the report, run `scripts/verify_fresh_clone.sh --fast` at the named commit, and sign below.

| Item | Verdict | Verifier | Date |
|---|---|---|---|
| Fast tier reproduced | | | |
| Full pipeline reproduced | | | |
| Discrepancies understood | | | |
| Sign-off | | | |
