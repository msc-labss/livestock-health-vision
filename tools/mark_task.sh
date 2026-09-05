#!/usr/bin/env bash
# Mark an OpenSpec task complete by its numbered prefix, e.g. tools/mark_task.sh 1.1
set -euo pipefail
TASKS="openspec/changes/add-lameness-spine-p0/tasks.md"
for id in "$@"; do
  python3 - "$TASKS" "$id" <<'PY'
import re, sys
path, task_id = sys.argv[1], sys.argv[2]
text = open(path).read()
pattern = re.compile(r"^- \[ \] (" + re.escape(task_id) + r") ", re.M)
new, n = pattern.subn(r"- [x] \1 ", text)
if n != 1:
    raise SystemExit(f"expected exactly one pending task {task_id!r}, matched {n}")
open(path, "w").write(new)
print(f"marked {task_id}")
PY
done
