#!/bin/bash
# Full Hebrew training on this Mac (MPS; DEVICE=cuda or cpu elsewhere). Resumes from last.pt after any exit.
# Usage: [ANNOTATIONS=work/annotations-v3] scripts/train_local.sh RUN [INIT_CHECKPOINT]
#   Stop cleanly:  touch RUN/STOP     Follow:  tail -f RUN.log
# The process restarts every 2 h (--max-seconds) so MPS memory cannot build up.
# gan-scale-max 10: the adaptive GAN scale normally sits at 1.5-6; in run he-v1 it
# spiked to the old 2000 cap at step 16350 and the generator collapsed.
cd "$(dirname "$0")/.."
RUN=${1:?run directory}; INIT=$2; LOG=$RUN.log; ANNOTATIONS=${ANNOTATIONS:-work/annotations}; DEVICE=${DEVICE:-mps}
while [ ! -f "$RUN/STOP" ]; do
  START=()
  if [ -f "$RUN/last.pt" ]; then START=(--resume "$RUN/last.pt")
  elif [ -n "$INIT" ]; then START=(--init-from "$INIT"); fi
  PYTHONPATH=src .venv/bin/python -m tiny_voca.train \
    --train $ANNOTATIONS/train.jsonl --validation $ANNOTATIONS/validation.jsonl \
    --output "$RUN" --device $DEVICE --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
    --gan-scale-max 10 --keep-every 2000 --no-preload \
    --max-seconds 7200 --stop-file "$RUN/STOP" "${START[@]}" 2>&1 | grep --line-buffered -v -i warn >> "$LOG"
  grep -q "already reached requested total steps" "$LOG" && break
  python3 -c "import json;d=[json.loads(l) for l in open('$RUN/metrics.jsonl')];exit(0 if d and d[-1].get('step',0)>=104000 else 1)" 2>/dev/null && break
  grep -q "output directory is not empty" "$LOG" && [ ! -f "$RUN/last.pt" ] && { echo "cannot start: $RUN not empty and no last.pt" >> "$LOG"; break; }
  sleep 10
done
echo "finished $(date)" >> "$LOG"
