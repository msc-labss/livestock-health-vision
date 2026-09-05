#!/usr/bin/env bash
# Fail if version control carries video, derived media or model weights.
#
# The repository states that none of these are ever committed. This turns that
# statement into something checkable, and checks it two ways: what is tracked
# now, and what a commit would add. Checking only the first would pass happily
# while a 40 GB dataset sat untracked and unignored, waiting for `git add .`.
set -euo pipefail
cd "$(dirname "$0")/.."

PATTERN='\.(mp4|avi|mkv|mov|webm|m4v|mpg|mpeg|pt|pth|onnx|engine|tflite|pb|safetensors|ckpt|weights|npz|parquet|duckdb)$'

# Everything version control would carry: tracked files, plus untracked files
# that nothing ignores.
candidates="$( { git ls-files; git ls-files --others --exclude-standard; } | sort -u )"

offenders="$(printf '%s\n' "$candidates" | grep -iE "$PATTERN" || true)"
if [[ -n "$offenders" ]]; then
  echo "version control carries media or model weights, which this repository forbids:" >&2
  printf '%s\n' "$offenders" | sed 's/^/  /' >&2
  exit 1
fi

large="$(printf '%s\n' "$candidates" | tr '\n' '\0' \
  | xargs -0 -r du -k 2>/dev/null \
  | awk '$1 > 20480 {print $2 " (" $1 " KiB)"}' || true)"
if [[ -n "$large" ]]; then
  echo "version control carries files over 20 MiB, which are assumed to be payloads:" >&2
  printf '%s\n' "$large" | sed 's/^/  /' >&2
  exit 1
fi

count="$(printf '%s\n' "$candidates" | grep -c . || true)"
echo "no video, derived media or model weights are carried by version control ($count file(s))"
