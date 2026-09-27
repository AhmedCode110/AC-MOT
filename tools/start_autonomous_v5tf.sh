#!/bin/zsh
# Start exactly one detached V5-TF continuation supervisor.
set -eu
cd /Users/ahmedgouda/Desktop/Universal-ACMOT

mkdir -p outputs/autonomous_v5tf
if [ -f outputs/autonomous_v5tf/repo_writer.lock/pid ]; then
  existing=$(<outputs/autonomous_v5tf/repo_writer.lock/pid)
  if kill -0 "$existing" 2>/dev/null; then
    echo "Supervisor already running: pid $existing"
    exit 0
  fi
fi

nohup .venv/bin/python -u tools/autonomous_v5tf_supervisor.py \
  >> outputs/autonomous_v5tf/launcher.log 2>&1 </dev/null &
echo $! > outputs/autonomous_v5tf/launcher.pid
echo "Started V5-TF autonomous supervisor: pid $!"
echo "Status: .venv/bin/python tools/autonomous_v5tf_supervisor.py --status"
echo "Log: outputs/autonomous_v5tf/supervisor.log"
