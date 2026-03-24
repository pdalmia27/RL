#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess

import ray
from ray._private.runtime_env import nsight


def sh(cmd: str) -> str:
    return subprocess.check_output(["bash", "-lc", cmd], text=True).strip()


@ray.remote
class InspectActor:
    def probe(self) -> dict[str, str]:
        return {
            "pid": str(os.getpid()),
            "argv": sh(f"ps -o args= -p {os.getpid()}"),
            "exe": os.readlink(f"/proc/{os.getpid()}/exe"),
            "path": os.environ.get("PATH", ""),
            "ray_nsys_binary": os.environ.get("RAY_NSYS_BINARY", ""),
            "nsight_systems_version_env": os.environ.get("NSIGHT_SYSTEMS_VERSION", ""),
            "which_nsys": sh("command -v nsys || true"),
            "nsys_version_default": sh("nsys --version | head -1 || true"),
            "nsys_version_usr_local": sh("/usr/local/bin/nsys --version | head -1 || true"),
        }


def main() -> int:
    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if ray_address:
        ray.init(address=ray_address, logging_level="ERROR")
    else:
        ray.init(logging_level="ERROR", include_dashboard=False)
    runtime_env = {
        "_nsight": {
            "t": "cuda,cudnn,cublas,nvtx",
            "o": "'actor_smoke_%p'",
            "stop-on-exit": "true",
        }
    }
    print("parse_default=" + " ".join(nsight.parse_nsight_config(runtime_env["_nsight"])))
    actor = InspectActor.options(runtime_env=runtime_env).remote()
    result = ray.get(actor.probe.remote())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
