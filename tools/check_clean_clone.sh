#!/usr/bin/env bash
# Verify that a clean clone contains no media, weights or dataset payload.
#
# A clone is the only way to see what someone else actually receives. Checking
# the working tree cannot: it is full of things that are correctly ignored.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

PATTERN='\.(mp4|avi|mkv|mov|webm|m4v|mpg|mpeg|pt|pth|onnx|engine|tflite|pb|safetensors|ckpt|weights|npz|parquet|duckdb)$'
CLONE="$(mktemp -d)"
trap 'rm -rf "$CLONE"' EXIT

git clone --quiet --no-hardlinks "$REPO" "$CLONE/repo"
rm -rf "$CLONE/repo/.git"

offenders="$(find "$CLONE/repo" -type f | sed "s|$CLONE/repo/||" | grep -iE "$PATTERN" || true)"
if [[ -n "$offenders" ]]; then
  echo "a clean clone contains media or model weights:" >&2
  printf '%s\n' "$offenders" | sed 's/^/  /' >&2
  exit 1
fi

large="$(find "$CLONE/repo" -type f -size +20M | sed "s|$CLONE/repo/||" || true)"
if [[ -n "$large" ]]; then
  echo "a clean clone contains files over 20 MiB:" >&2
  printf '%s\n' "$large" | sed 's/^/  /' >&2
  exit 1
fi

files="$(find "$CLONE/repo" -type f | wc -l | tr -d ' ')"
bytes="$(du -sh "$CLONE/repo" | cut -f1)"
echo "a clean clone contains no media, weights or dataset payload ($files file(s), $bytes)"
