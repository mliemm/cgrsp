.PHONY: help proofread proofread-quick lint spell grammar validate test clean

help:
	@echo "CGRSP Thesis targets:"
	@echo "  proofread           Run full grammar, spelling, and LaTeX linting (CheckMyTex + TeXtidote)"
	@echo "  proofread-quick     Run spell check only (codespell)"
	@echo "  grammar             Run TeXtidote for grammar/style checking"
	@echo "  spell               Run codespell for spell checking"
	@echo "  lint                Run ChkTeX LaTeX structural linting"
	@echo "  validate            Run Python validation (validate_numbers.py, codespell)"
	@echo "  test                Run Python tests"
	@echo "  clean               Remove build artifacts"

# Full proofreading: grammar, spell, and LaTeX linting via CheckMyTex (unified)
proofread:
	@echo "=== CheckMyTex: grammar + spelling + ChkTeX + proselint ==="
	checkmytex thesis/thesis.tex
	@echo ""
	@echo "=== TeXtidote: additional grammar/style pass ==="
	java -jar ~/.local/share/textidote/TeXtidote.jar --check en thesis/thesis.tex || true

# Quick spell check only
proofread-quick:
	@echo "=== Codespell: spell check ==="
	codespell

# Grammar and style via TeXtidote (requires Java)
grammar:
	@echo "=== TeXtidote: grammar and style checking ==="
	java -jar ~/.local/share/textidote/TeXtidote.jar --check en thesis/thesis.tex

# Spell checking via codespell
spell:
	@echo "=== Codespell: spell check ==="
	codespell

# LaTeX linting via ChkTeX
lint:
	@echo "=== ChkTeX: LaTeX structural linting ==="
	chktex thesis/thesis.tex

# Python validation (numbers, consistency)
validate:
	@echo "=== Validating generated data and macros ==="
	python3 validate_numbers.py
	@echo ""
	@echo "=== Codespell: spell check (thesis only) ==="
	codespell thesis/

# Python tests
# The interpreter that produces the results (python3, 3.14) has no pytest and
# refuses installation under PEP 668, so the suite runs on python3.12.
# Override with e.g. PYTEST_PYTHON=python3.13 make test
PYTEST_PYTHON ?= python3.12

test:
	@echo "=== Running Python tests ($(PYTEST_PYTHON)) ==="
	$(PYTEST_PYTHON) -m pytest

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f thesis/texput.log
	@echo "Cleaned."
