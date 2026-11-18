#!/bin/bash
# From the Mac, while the pod trains: every 15 min pull RUN (checkpoints, metrics)
# and RUN.log into this checkout. When the pod marks RUN/DONE, pull once more and
# terminate the pod; on RUN/FAILED pull and stop it (volume kept for debugging).
# Usage: caffeinate -is scripts/runpod/sync.sh [RUN]
# Exit: 0 done and terminated, 2 failed and stopped, 3 pod unreachable for 2 h.
cd "$(dirname "$0")/../.."
. scripts/runpod/common.sh
RUN=${1:-runs/he-v3}; MISSES=0
pull() {
  rsync -az --partial --exclude STOP -e "$RSYNC_SSH" "$REMOTE/$RUN/" "$RUN/" && \
  rsync -az -e "$RSYNC_SSH" "$REMOTE/$RUN.log" "$RUN.log"
}
while true; do
  if pod_connect 120 && pull; then
    MISSES=0
    python3 -c "import json;d=json.loads(open('$RUN/metrics.jsonl').readlines()[-1]);print('$(date +%H:%M)', 'step', d.get('step'), 'mel', round(d.get('mel', 0), 3), 'gan_scale', round(d.get('gan_scale') or 0, 2))"
    if [ -f "$RUN/DONE" ]; then pull; python3 scripts/runpod/pod.py terminate; echo "done $(date)"; exit 0; fi
    if [ -f "$RUN/FAILED" ]; then python3 scripts/runpod/pod.py stop; echo "failed $(date)"; exit 2; fi
  else
    MISSES=$((MISSES + 1)); echo "$(date +%H:%M) pod unreachable ($MISSES)"
    [ "$MISSES" -ge 8 ] && exit 3
  fi
  sleep 900
done
