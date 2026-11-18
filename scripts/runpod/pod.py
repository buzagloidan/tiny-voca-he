"""Minimal RunPod control for the GPU training run (GraphQL API, stdlib only).

    python3 scripts/runpod/pod.py create [--gpu "NVIDIA GeForce RTX 4090"] [--cloud COMMUNITY]
    python3 scripts/runpod/pod.py address      # prints "IP PORT" of the pod's SSH, waits until it is up
    python3 scripts/runpod/pod.py info         # status, price, uptime
    python3 scripts/runpod/pod.py stop         # stop billing for the GPU, keep the volume
    python3 scripts/runpod/pod.py start        # resume a stopped pod (needs its GPU to be free)
    python3 scripts/runpod/pod.py terminate    # delete the pod and its volume
    python3 scripts/runpod/pod.py balance

The API key comes from RUNPOD_API_KEY or the file RUNPOD_ENV_FILE (default
~/.runpod/tiny-voca.env), never from the repository. The pod id is kept in
runs/runpod-pod.json. SSH uses the key pair RUNPOD_SSH_KEY (default
~/.ssh/runpod_tiny_voca); its public half is handed to the pod as PUBLIC_KEY.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

API = "https://api.runpod.io/graphql"
STATE = Path(__file__).resolve().parents[2] / "runs" / "runpod-pod.json"
SSH_KEY = Path(os.environ.get("RUNPOD_SSH_KEY", Path.home() / ".ssh" / "runpod_tiny_voca"))
# Only the SSH setup of this image is used; setup_pod.sh installs its own torch.
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    env = Path(os.environ.get("RUNPOD_ENV_FILE", Path.home() / ".runpod" / "tiny-voca.env"))
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(f"no RunPod API key: set RUNPOD_API_KEY or write RUNPOD_API_KEY=... into {env}")
    return key


def gql(query, variables=None):
    # curl, not urllib: Python 3.13+ strict X.509 checks reject the API's certificate chain.
    # The key goes in through stdin so it never shows in the process list.
    body = json.dumps({"query": query, "variables": variables or {}})
    config = (f'header = "Authorization: Bearer {api_key()}"\nheader = "Content-Type: application/json"\n'
              f'user-agent = "tiny-voca-runpod"\nurl = "{API}"\n')
    with tempfile.NamedTemporaryFile("w", suffix=".json") as data:
        data.write(body)
        data.flush()
        result = subprocess.run(["curl", "-sS", "-m", "60", "--fail-with-body", "-K", "-", "--data-binary", f"@{data.name}"],
                                input=config, capture_output=True, text=True)
    if result.returncode and not result.stdout:
        raise RuntimeError(result.stderr.strip())
    payload = json.loads(result.stdout)
    if payload.get("errors"):
        raise RuntimeError("; ".join(e.get("message", str(e)) for e in payload["errors"]))
    return payload["data"]


def pod_id():
    if not STATE.exists():
        sys.exit(f"no pod recorded in {STATE}")
    return json.loads(STATE.read_text())["id"]


def pod_info(identifier):
    return gql("""query($input: PodFilter!) { pod(input: $input) { id name desiredStatus costPerHr lastStatusChange
        machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type }
        gpus { gpuUtilPercent memoryUtilPercent } } } }""", {"input": {"podId": identifier}})["pod"]


def create(args):
    if STATE.exists():
        sys.exit(f"a pod is already recorded in {STATE}; terminate it first")
    public_key = (SSH_KEY.with_suffix(".pub")).read_text().strip()
    last_error = None
    for vcpus, memory in ((8, 30), (6, 24), (4, 16)):
        settings = {"name": "tiny-voca-he", "imageName": IMAGE, "gpuTypeId": args.gpu, "cloudType": args.cloud,
                    "gpuCount": 1, "volumeInGb": 20, "containerDiskInGb": 30, "volumeMountPath": "/workspace",
                    "minVcpuCount": vcpus, "minMemoryInGb": memory, "ports": "22/tcp", "supportPublicIp": True,
                    "startSsh": True, "dockerArgs": "", "env": [{"key": "PUBLIC_KEY", "value": public_key}]}
        try:
            pod = gql("""mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) {
                id costPerHr machine { gpuDisplayName } } }""", {"input": settings})["podFindAndDeployOnDemand"]
        except RuntimeError as error:
            last_error = error
            print(f"no {args.gpu} with {vcpus} vCPU / {memory} GB: {error}", file=sys.stderr)
            continue
        STATE.write_text(json.dumps({**pod, "created": time.time(), "vcpus": vcpus, "memory_gb": memory}, indent=2))
        print(json.dumps(pod))
        return
    sys.exit(f"could not deploy a pod: {last_error}")


def address(args):
    identifier, deadline = pod_id(), time.time() + args.timeout
    while time.time() < deadline:
        pod = pod_info(identifier)
        if pod is None:
            sys.exit("pod no longer exists")
        for port in ((pod.get("runtime") or {}).get("ports") or []):
            if port["privatePort"] == 22 and port["isIpPublic"]:
                print(port["ip"], port["publicPort"])
                return
        time.sleep(10)
    sys.exit("timed out waiting for the pod's SSH port")


def info(args):
    print(json.dumps(pod_info(pod_id()), indent=2))


def stop(args):
    print(json.dumps(gql("mutation($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
                         {"input": {"podId": pod_id()}})))


def start(args):
    print(json.dumps(gql("mutation($input: PodResumeInput!) { podResume(input: $input) { id desiredStatus costPerHr } }",
                         {"input": {"podId": pod_id(), "gpuCount": 1}})))


def terminate(args):
    gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": pod_id()}})
    STATE.rename(STATE.with_suffix(".terminated.json"))
    print("terminated")


def balance(args):
    print(json.dumps(gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    make.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
    make.set_defaults(run=create)
    wait = commands.add_parser("address")
    wait.add_argument("--timeout", type=float, default=900)
    wait.set_defaults(run=address)
    for name, function in (("info", info), ("stop", stop), ("start", start), ("terminate", terminate), ("balance", balance)):
        commands.add_parser(name).set_defaults(run=function)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
