#!/usr/bin/env bash
# Build the thesis PDF and update the public copy from the finished build.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
thesis_dir="$repo_root/thesis"
public_dir="$repo_root/public"

(
  cd "$thesis_dir"
  xelatex -interaction=nonstopmode -halt-on-error thesis.tex > thesis-build-pass1.log
  bibtex thesis > thesis-build-bibtex.log
  for pass in 2 3 4; do
    xelatex -interaction=nonstopmode -halt-on-error thesis.tex > "thesis-build-pass${pass}.log"
  done
)

mkdir -p "$public_dir"
cp "$thesis_dir/thesis.pdf" "$public_dir/thesis.pdf"
cmp "$thesis_dir/thesis.pdf" "$public_dir/thesis.pdf"
echo "Built thesis/thesis.pdf and copied it to public/thesis.pdf"
