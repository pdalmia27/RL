#!/usr/bin/env python3
from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - debug helper
        return f"<failed: {exc}>"


def main() -> int:
    from ray._private.runtime_env import nsight

    print(f"python={sys.executable}")
    print(f"nsight_module={nsight.__file__}")
    print(f"PATH={os.environ.get('PATH', '')}")
    print(f"RAY_NSYS_BINARY={os.environ.get('RAY_NSYS_BINARY', '')}")
    print(f"which_nsys={shutil.which('nsys')}")
    print(f"which_a_nsys={run(['bash', '-lc', 'which -a nsys || true'])}")
    print(f"nsys_version_default={run(['bash', '-lc', 'nsys --version | head -1'])}")
    print(
        f"nsys_version_usr_local={run(['bash', '-lc', '/usr/local/bin/nsys --version | head -1'])}"
    )
    print(
        "parse_default="
        + " ".join(
            nsight.parse_nsight_config(
                {"t": "cuda,cudnn,cublas,nvtx", "o": "'inspect_%p'", "stop-on-exit": "true"}
            )
        )
    )
    src = inspect.getsource(nsight.parse_nsight_config)
    print("parse_source_begin")
    print(src.rstrip())
    print("parse_source_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
