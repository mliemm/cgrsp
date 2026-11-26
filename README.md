# CGRSP

Cloud GPU Rendering Scheduling Problem (CGRSP): A Mathematical Formulation and Empirical Evaluation on Heterogeneous GPU Fleets.

Author: Michael (林墾森), M11215818, Department of Computer Science and Information Engineering, National Taiwan University of Science and Technology.

Advisors: Prof. Yuan-Shin Hwang (黃元欣) and Prof. Vincent F. Yu (喻奉天).

This repository contains the thesis source, the discrete-event simulator, the calibration data, the generated results, and the reproducibility package. The authoritative document is `thesis/thesis.pdf`; the source is `thesis/thesis.tex`.

## Quick start

```bash
# Rebuild the PDF from the committed tables and figures (about 1 minute)
./scripts/build_thesis.sh

# Validate every reported number against the generated macros and JSON
python3 validate_numbers.py

# Run the test suite
make test

# Regenerate every experiment, table, figure, and the PDF (about 1 hour)
./run_all.sh
```

`run_all.sh` is deterministic: every experiment entry point imports `CGRSP/_repro.py`, which pins `PYTHONHASHSEED=0`, and the provisioning-delay seed uses `zlib.crc32` rather than the interpreter hash.

## Package map

| Path | Contents |
|---|---|
| `thesis/` | LaTeX source, front matter, chapters, figures, generated tables, and the built PDF. |
| `CGRSP/` | Simulator, schedulers, experiment drivers, tests, and JSON outputs. |
| `CGRSP/Outputs/phase3/` | Phase 3 results, including `baselines.json` and `fleet_mix.json` for the baseline and fleet-composition tables. |
| `data/` | 6,234-job calibration dataset, excluded rows, and the compact pricing summary. |
| `thesis/generated/` | Generated statistics macros and table fragments. Do not edit by hand. |
| `scripts/` | Build, filtering, and packaging helpers. |
| `docs/` | Automated verification report, the output hash manifest, and `thesis_revisions_highlighted.pdf` (the final thesis with every change since the 9 November 2026 defense marked: additions in blue, deletions in red). |
| `REPRODUCE.md` | The full reproducibility package. |

## Data and code availability

The empirical dataset is included for reproducibility. It contains job timestamps, GPU type, render time, numeric material and model identifiers from the author's pipeline, and environment tags; it contains no client names or billing details. The raw RunPod pricing snapshot archive is not redistributed because of its size. `data/runpod_pricing_summary/` tracks the minimum columns needed to regenerate the two pricing figures, and `scripts/export_pricing_summary.py` rebuilds it from the archive.

This repository is a clean snapshot of the thesis package for review. The thesis history, defense material and correspondence are kept in a separate private repository.

## License

The code is released under the MIT License; see `LICENSE`. The thesis text, PDF, and empirical dataset are copyright 2026 Michael and are provided for academic reproducibility. The simulator core adapts SPARS, which is MIT-licensed; see `CGRSP/THIRD_PARTY_NOTICES.md`.

## Citation

Until a journal version appears, cite the thesis:

Michael. *Cloud GPU Rendering Scheduling Problem (CGRSP): A Mathematical Formulation and Empirical Evaluation on Heterogeneous GPU Fleets.* Master's thesis, National Taiwan University of Science and Technology, 2027.
