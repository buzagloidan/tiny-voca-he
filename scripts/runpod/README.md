# Training on RunPod

These scripts train on a rented RunPod GPU (from scratch, or continuing a run stopped locally with `touch RUN/STOP`) and bring the checkpoints back to your machine.

| Script | Runs on | What it does |
|---|---|---|
| `pod.py` | Mac | Creates, inspects, stops and terminates the pod through RunPod's GraphQL API. It also reports the account balance. |
| `launch.sh [RUN]` | Mac | Creates the pod unless one is recorded in `runs/runpod-pod.json`. It uploads the code, `work/annotations`, the train and validation features, and RUN, runs `setup_pod.sh`, then starts `train_gpu.sh` detached. |
| `setup_pod.sh` | pod | Installs uv and a `.venv` with a CUDA build of torch that matches the host driver. It links the Mac path recorded as `acoustic_root` to `/workspace/tiny-voca-he`. |
| `train_gpu.sh RUN` | pod | Uses the same recipe as `scripts/train_local.sh`, with `--device cuda`, the data preloaded, and restarts every 6 h. It marks `RUN/DONE` or `RUN/FAILED`. |
| `sync.sh [RUN]` | Mac | Pulls RUN and `RUN.log` every 15 min. On DONE it terminates the pod; on FAILED it stops the pod. |

**Setup**
- Put the API key in `RUNPOD_API_KEY`, or in `~/.runpod/tiny-voca.env` as `RUNPOD_API_KEY=...` (another file: `RUNPOD_ENV_FILE`). Never commit it.
- SSH uses `~/.ssh/runpod_tiny_voca` (another key: `RUNPOD_SSH_KEY`). Create it with `ssh-keygen -t ed25519 -f ~/.ssh/runpod_tiny_voca -N ''`. Its public key is passed to the pod as `PUBLIC_KEY`.

**Typical use**

```bash
scripts/runpod/launch.sh runs/he-v3
caffeinate -is scripts/runpod/sync.sh runs/he-v3
```

**Billing safety nets**
- `train_gpu.sh` stops the pod 2 h after the run ends.
- The pod also stops itself after 36 h in total.
- Both need `runpodctl` and `RUNPOD_POD_ID` on the pod; RunPod images provide them.
- A stopped pod still bills its 20 GB volume until you run `pod.py terminate`.
