#!/bin/zsh
# Ask the supervisor to stop safely. It also stops its active Codex child.
set -eu
cd /Users/ahmedgouda/Desktop/Universal-ACMOT

pid_file=outputs/autonomous_v5tf/repo_writer.lock/pid
if [ ! -f "$pid_file" ]; then
  echo "No autonomous V5-TF supervisor lock is present."
  exit 0
fi

pid=$(<"$pid_file")
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Supervisor pid $pid is not running; lock is stale. Start command will recover it."
  exit 0
fi

command=$(ps -p "$pid" -o command= 2>/dev/null || true)
case "$command" in
  *tools/autonomous_v5tf_supervisor.py*) ;;
  *)
    echo "Refusing to signal pid $pid because it is not the V5-TF supervisor."
    exit 1
    ;;
esac

echo "Requesting safe stop for supervisor pid $pid..."
kill -TERM "$pid"
for _ in {1..30}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Supervisor stopped cleanly."
    exit 0
  fi
  sleep 1
done

echo "Supervisor is still stopping. No force-kill was issued; check status/logs."
exit 2

