#!/bin/zsh
# Human-readable status for the V5-TF continuation supervisor.
set -u
cd /Users/ahmedgouda/Desktop/Universal-ACMOT

.venv/bin/python tools/autonomous_v5tf_supervisor.py --status
if [ -f outputs/autonomous_v5tf/repo_writer.lock/pid ]; then
  pid=$(<outputs/autonomous_v5tf/repo_writer.lock/pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "lock: held by live supervisor pid $pid"
  else
    echo "lock: stale pid $pid"
  fi
else
  echo "lock: not held"
fi

echo "recent log:"
tail -20 outputs/autonomous_v5tf/supervisor.log 2>/dev/null || true

