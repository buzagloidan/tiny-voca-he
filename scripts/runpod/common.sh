# Sourced by launch.sh and sync.sh (Mac side). Sets IP, PORT, SSH, RSYNC_SSH for the recorded pod.
pod_connect() {
  read -r IP PORT < <(python3 scripts/runpod/pod.py address --timeout "${1:-900}") || return 1
  local options="-i ${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_tiny_voca} -p $PORT -o UserKnownHostsFile=$HOME/.ssh/runpod_known_hosts \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes"
  SSH=(ssh $options root@"$IP")
  RSYNC_SSH="ssh $options"
  REMOTE=root@$IP:/workspace/tiny-voca-he
}
