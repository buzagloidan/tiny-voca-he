#!/bin/bash
# From the Mac: create the pod (unless one is recorded), upload code, features and
# the run, install the GPU environment and start training there detached.
# Usage: [FEATURES=work/features ANNOTATIONS=work/annotations] scripts/runpod/launch.sh [RUN]
#   (default runs/he-v3). RUN/last.pt is resumed; without it RUN must be new or empty
#   and the run starts from scratch (or from INIT_FROM set in RUN.args, see train_gpu.sh).
# Then: scripts/runpod/sync.sh RUN  pulls checkpoints and terminates the pod at the end.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/runpod/common.sh
RUN=${1:-runs/he-v3}; FEATURES=${FEATURES:-work/features}; ANNOTATIONS=${ANNOTATIONS:-work/annotations}
INIT_FROM=; ARGS=()
[ -f "$RUN.args" ] && { . "$RUN.args"; ARGS=("$RUN.args"); }
[ -z "$INIT_FROM" ] || [ -f "$INIT_FROM" ] || { echo "missing $INIT_FROM" >&2; exit 1; }
if [ ! -f "$RUN/last.pt" ]; then
  [ -z "$(ls -A "$RUN" 2>/dev/null)" ] || { echo "$RUN has files but no last.pt" >&2; exit 1; }
  echo "no $RUN/last.pt: starting $RUN from scratch"; mkdir -p "$RUN"; touch "$RUN.log"
fi
[ -f runs/runpod-pod.json ] || python3 scripts/runpod/pod.py create
pod_connect
echo "pod SSH at $IP:$PORT"
for attempt in $(seq 60); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
"${SSH[@]}" "mkdir -p /workspace/tiny-voca-he && (command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null))"

echo "uploading $(date +%T)"
rsync -azR --no-owner --no-group --partial --exclude __pycache__ --exclude STOP --exclude DONE --exclude FAILED -e "$RSYNC_SSH" \
  pyproject.toml LICENSE src scripts \
  "$ANNOTATIONS/annotations.json" "$ANNOTATIONS/train.jsonl" "$ANNOTATIONS/validation.jsonl" \
  "$FEATURES"/*.json "$FEATURES"/*.jsonl \
  "$RUN" "$RUN.log" ${ARGS[@]+"${ARGS[@]}"} ${INIT_FROM:+"$INIT_FROM"} "$REMOTE/"
# Only the feature files the annotations accepted, over 16 parallel connections:
# a single SSH stream to RunPod hosts reached 0.16-3 MB/s from here, 16 together ~16 MB/s.
LISTS=$(mktemp -d)
python3 - "$LISTS" "$ANNOTATIONS/train.jsonl" "$ANNOTATIONS/validation.jsonl" <<'PY'
import json, sys
paths = [json.loads(line)["path"] for name in sys.argv[2:] for line in open(name)]
for i in range(16):
    open(f"{sys.argv[1]}/part.{i:02d}", "w").write("".join(p + "\n" for p in paths[i::16]))
PY
for part in "$LISTS"/part.*; do
  # sshd may drop some of 16 simultaneous logins; each part retries.
  ( for try in 1 2 3 4 5; do rsync -a --no-owner --no-group --partial --files-from="$part" -e "$RSYNC_SSH" "$FEATURES/" "$REMOTE/$FEATURES/" && exit 0; sleep $((try * 5)); done; exit 1 ) &
done
FAILED_PARTS=0
for job in $(jobs -p); do wait "$job" || FAILED_PARTS=$((FAILED_PARTS + 1)); done
[ "$FAILED_PARTS" = 0 ] || { echo "$FAILED_PARTS upload parts failed" >&2; exit 1; }
rm -r "$LISTS"
echo "uploaded $(date +%T)"

"${SSH[@]}" "cd /workspace/tiny-voca-he && bash scripts/runpod/setup_pod.sh $ANNOTATIONS"
# Only the trainer goes to the background (not the whole && list), so the SSH session can close.
"${SSH[@]}" "cd /workspace/tiny-voca-he && rm -f $RUN/STOP $RUN/DONE $RUN/FAILED; \
  nohup setsid bash scripts/runpod/train_gpu.sh $RUN $ANNOTATIONS > /dev/null 2>&1 < /dev/null & echo started"
