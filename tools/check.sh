#!/usr/bin/env bash
# The project check harness: lint, species seam, tests.
#
# PYTHONPATH is cleared deliberately. A sourced ROS (or any other) environment
# puts a foreign site-packages on sys.path, which drags foreign pytest plugins
# into this project's interpreter. The project must be checked against its own
# resolved environment and nothing else.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=""
PY="${PY:-.venv/bin/python}"

echo "== ruff format =="
"$PY" -m ruff format --check src tests tools

echo "== ruff lint =="
"$PY" -m ruff check src tests tools

echo "== species seam =="
"$PY" tools/check_species_seam.py

echo "== profile declarations =="
"$PY" tools/check_profile_declarations.py

echo "== tests =="
"$PY" -m pytest "$@"

echo "== all checks passed =="
