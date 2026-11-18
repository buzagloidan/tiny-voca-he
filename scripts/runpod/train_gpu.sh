#!/bin/bash
# Runs ON the pod. Same recipe as scripts/train_local.sh, on CUDA, with the data
# preloaded into memory. Resumes RUN/last.pt after any exit.
# Usage (detached): nohup setsid scripts/runpod/train_gpu.sh runs/he-v3 [ANNOTATIONS] > /dev/null 2>&1 &
# Without RUN/last.pt the first start trains from scratch, or branches from
# INIT_FROM when RUN.args sets it. RUN.args (optional, bash) may set STEPS,
# INIT_FROM and TRAIN_ARGS=(...), extra trainer options given on every start.
# Stop cleanly: touch RUN/STOP. Marks RUN/DONE when the run reaches its last step,
# RUN/FAILED if it cannot continue.
# Safety net for billing: the pod stops itself 2 h after DONE/FAILED (the Mac's
# sync.sh normally pulls the results and terminates it first) and after 36 h in all.
cd "$(dirname "$0")/../.."
RUN=${1:?run directory}; ANNOTATIONS=${2:-work/annotations}; LOG=$RUN.log; STEPS=104000; INIT_FROM=; TRAIN_ARGS=()
[ -f "$RUN.args" ] && . "$RUN.args"

# SSH sessions do not inherit the container environment; take the pod id and its
# pod-scoped key for runpodctl from PID 1.
eval "$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^RUNPOD_(POD_ID|API_KEY)=[A-Za-z0-9_-]+$' | sed 's/^/export /')"
stop_pod_after() {  # seconds
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null || return 0
  nohup setsid bash -c "sleep $1; runpodctl stop pod $RUNPOD_POD_ID" > /dev/null 2>&1 &
}
stop_pod_after $((36 * 3600))

while [ ! -f "$RUN/STOP" ]; do
  START=(--resume "$RUN/last.pt")
  if [ ! -f "$RUN/last.pt" ]; then
    [ -n "$STARTED" ] && { echo "no $RUN/last.pt to resume" >> "$LOG"; touch "$RUN/FAILED"; break; }
    START=()
    [ -n "$INIT_FROM" ] && START=(--init-from "$INIT_FROM" --new-data)
  fi
  STARTED=1
  PYTHONPATH=src .venv/bin/python -m tiny_voca.train \
    --train $ANNOTATIONS/train.jsonl --validation $ANNOTATIONS/validation.jsonl \
    --output "$RUN" --device cuda --steps $STEPS --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
    --gan-scale-max 10 --keep-every 2000 \
    --max-seconds 21600 --stop-file "$RUN/STOP" "${TRAIN_ARGS[@]}" "${START[@]}" 2>&1 | grep --line-buffered -v -i warn >> "$LOG"
  if grep -q "already reached requested total steps" "$LOG" || \
     python3 -c "import json;d=[json.loads(l) for l in open('$RUN/metrics.jsonl')];exit(0 if d and d[-1].get('step',0)>=$STEPS else 1)" 2>/dev/null; then
    touch "$RUN/DONE"; break
  fi
  FAILS=$(( ${FAILS:-0} + 1 ))
  [ "$FAILS" -ge 5 ] && { echo "giving up after $FAILS failed starts" >> "$LOG"; touch "$RUN/FAILED"; break; }
  # A clean --max-seconds exit is not a failure.
  tail -n 5 "$LOG" | grep -q -i -E "error|traceback" || FAILS=0
  sleep 10
done
echo "finished $(date)" >> "$LOG"
[ -f "$RUN/STOP" ] || stop_pod_after 7200
