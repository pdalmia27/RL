#!/usr/bin/env python3
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
import re


RAY_TARGET_GLOBS = [
    "/opt/nemo_rl_venv/lib/python*/site-packages/ray/_private/runtime_env/nsight.py",
    "/opt/nemo_rl_venv/lib64/python*/site-packages/ray/_private/runtime_env/nsight.py",
]

VLLM_TARGET_GLOBS = [
    "/opt/nemo_rl_venv/lib/python*/site-packages/vllm/executor/ray_distributed_executor.py",
    "/opt/nemo_rl_venv/lib64/python*/site-packages/vllm/executor/ray_distributed_executor.py",
    "/usr/local/lib/python*/site-packages/vllm/executor/ray_distributed_executor.py",
    "/usr/local/lib/python*/dist-packages/vllm/executor/ray_distributed_executor.py",
]

RAY_MODULE_CANDIDATES = [
    "ray._private.runtime_env.nsight",
]

VLLM_MODULE_CANDIDATES = [
    "vllm.executor.ray_distributed_executor",
    "vllm.v1.executor.ray_distributed_executor",
]


def _expand_globs(patterns: list[str]) -> list[Path]:
    matched: list[Path] = []
    for pattern in patterns:
        matched.extend(Path(match) for match in glob.glob(pattern))
    return sorted(set(matched))


def _resolve_modules(module_names: list[str]) -> list[Path]:
    matched: list[Path] = []
    for module_name in module_names:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            continue
        if spec and spec.origin and spec.origin not in {"built-in", "frozen"}:
            matched.append(Path(spec.origin))
    return sorted(set(matched))


def _collect_targets(globs: list[str], modules: list[str]) -> list[Path]:
    return sorted(set(_resolve_modules(modules) + _expand_globs(globs)))


def patch_ray_nsight(path: Path) -> bool:
    text = path.read_text()
    original = text

    text = text.replace(
        '    nsight_cmd = ["nsys", "profile"]\n',
        '    nsight_bin = os.environ.get("RAY_NSYS_BINARY", "nsys")\n'
        '    nsight_cmd = [nsight_bin, "profile"]\n'
        '    if nsight_config is None:\n'
        '        nsight_config = {}\n'
        '    else:\n'
        '        nsight_config = dict(nsight_config)\n'
        '    trace_modules = [module.strip() for module in nsight_config.get("t", "").split(",") if module.strip()]\n'
        '    if trace_modules:\n'
        '        if "nvtx" not in trace_modules:\n'
        '            trace_modules.append("nvtx")\n'
        '        nsight_config["t"] = ",".join(trace_modules)\n'
        '    else:\n'
        '        nsight_config["t"] = "cuda,cudnn,cublas,nvtx"\n'
        '    nsight_config.setdefault("capture-range", "cudaProfilerApi")\n'
        '    nsight_config.setdefault("capture-range-end", "stop")\n'
        '    nsight_config.setdefault("stop-on-exit", "true")\n',
    )
    text = text.replace(
        '        logger.info("Running nsight profiler")\n'
        '        context.py_executable = " ".join(self.nsight_cmd) + " python"\n',
        '        logger.info(f"Running nsight profiler via {\' \'.join(self.nsight_cmd)}")\n'
        '        context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"\n',
    )

    if text != original:
        path.write_text(text)
        return True
    return False


def patch_vllm_ray_executor_nsight(path: Path) -> bool:
    text = path.read_text()
    original = text

    old_block = """        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas",
                "o": "'worker_process_%p'",
                "cuda-graph-trace": "node",
            }
        })
"""
    new_block = """        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas,nvtx",
                "o": "'worker_process_%p'",
                "stop-on-exit": "true",
                "capture-range": "cudaProfilerApi",
                "capture-range-end": "stop",
                "cuda-graph-trace": "node",
            }
        })
"""

    text = text.replace(old_block, new_block)
    if text == original:
        text = re.sub(
            r'(?ms)^([ \t]*)runtime_env\.update\(\{\n\1    "nsight": \{\n.*?\n\1    \}\n\1\}\)\n',
            new_block,
            text,
            count=1,
        )

    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    ray_targets = _collect_targets(RAY_TARGET_GLOBS, RAY_MODULE_CANDIDATES)
    if not ray_targets:
        raise SystemExit(f"no nsight.py matched any of: {RAY_TARGET_GLOBS}")

    ray_changed = False
    for target in ray_targets:
        ray_changed = patch_ray_nsight(target) or ray_changed

    vllm_targets = _collect_targets(VLLM_TARGET_GLOBS, VLLM_MODULE_CANDIDATES)
    vllm_changed = False
    for target in vllm_targets:
        vllm_changed = patch_vllm_ray_executor_nsight(target) or vllm_changed

    print(
        f"patched_ray={ray_changed} matched_ray={len(ray_targets)} "
        f"patched_vllm={vllm_changed} matched_vllm={len(vllm_targets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
