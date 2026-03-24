# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import logging
import os
from copy import deepcopy
from typing import Any

from nemo_rl.utils.nsys import NRL_NSYS_PROFILE_STEP_RANGE, NRL_NSYS_WORKER_PATTERNS

NRL_NSYS_VLLM_WORKER_INDICES_ENV = "NRL_NSYS_VLLM_WORKER_INDICES"
NRL_RAY_WORKER_IDX_ENV = "NRL_RAY_WORKER_IDX"

_VLLM_NSYS_WORKER_NAMES = frozenset(
    {"vllm_generation_worker", "vllm_async_generation_worker"}
)


def _get_profiled_vllm_worker_indices() -> set[int] | None:
    worker_indices_env = os.environ.get(NRL_NSYS_VLLM_WORKER_INDICES_ENV, "").strip()
    if not worker_indices_env:
        return None

    worker_indices: set[int] = set()
    for worker_index_str in worker_indices_env.split(","):
        worker_index_str = worker_index_str.strip()
        if not worker_index_str:
            continue
        try:
            worker_index = int(worker_index_str)
        except ValueError as exc:
            raise ValueError(
                f"{NRL_NSYS_VLLM_WORKER_INDICES_ENV} must be a comma-separated list of integers, "
                f"got {worker_indices_env!r}"
            ) from exc

        if worker_index < 0:
            raise ValueError(
                f"{NRL_NSYS_VLLM_WORKER_INDICES_ENV} only supports non-negative worker indices, "
                f"got {worker_index}"
            )
        worker_indices.add(worker_index)
    return worker_indices


def _resolve_worker_idx(worker_idx: int | None) -> int | None:
    if worker_idx is not None:
        return worker_idx

    worker_idx_env = os.environ.get(NRL_RAY_WORKER_IDX_ENV, "").strip()
    if not worker_idx_env:
        return None

    try:
        resolved_worker_idx = int(worker_idx_env)
    except ValueError as exc:
        raise ValueError(
            f"{NRL_RAY_WORKER_IDX_ENV} must be an integer when set, got {worker_idx_env!r}"
        ) from exc

    if resolved_worker_idx < 0:
        raise ValueError(
            f"{NRL_RAY_WORKER_IDX_ENV} only supports non-negative worker indices, got {resolved_worker_idx}"
        )

    return resolved_worker_idx


def is_vllm_worker_nsight_index_filter_enabled() -> bool:
    return _get_profiled_vllm_worker_indices() is not None


def get_nsight_worker_name_for_actor_class(ray_actor_class_fqn: str) -> str | None:
    class_name = ray_actor_class_fqn.rsplit(".", 1)[-1]
    if class_name == "VllmGenerationWorker":
        return "vllm_generation_worker"
    if class_name == "VllmAsyncGenerationWorker":
        return "vllm_async_generation_worker"
    return None


def get_nsight_config_if_pattern_matches(
    worker_name: str, worker_idx: int | None = None
) -> dict[str, Any]:
    """Check if worker name matches patterns in NRL_NSYS_WORKER_PATTERNS and return nsight config.

    Args:
        worker_name: Name of the worker to check against patterns
        worker_idx: Optional stable worker index used to restrict profiling to a
            subset of vLLM workers when NRL_NSYS_VLLM_WORKER_INDICES is set.

    Returns:
        Dictionary containing {"nsight": config} if pattern matches, empty dict otherwise
    """
    assert not (bool(NRL_NSYS_WORKER_PATTERNS) ^ bool(NRL_NSYS_PROFILE_STEP_RANGE)), (
        "Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set, or neither. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
    )

    patterns_env = NRL_NSYS_WORKER_PATTERNS
    if not patterns_env:
        return {}

    # Parse CSV patterns
    patterns = [
        pattern.strip() for pattern in patterns_env.split(",") if pattern.strip()
    ]
    profiled_vllm_worker_indices = _get_profiled_vllm_worker_indices()
    resolved_worker_idx = _resolve_worker_idx(worker_idx)

    # Check if worker name matches any pattern
    for pattern in patterns:
        if fnmatch.fnmatch(worker_name, pattern):
            if (
                profiled_vllm_worker_indices is not None
                and worker_name in _VLLM_NSYS_WORKER_NAMES
            ):
                if resolved_worker_idx is None:
                    return {}
                if resolved_worker_idx not in profiled_vllm_worker_indices:
                    return {}

                output_name = (
                    f"'{worker_name}_{NRL_NSYS_PROFILE_STEP_RANGE}_w{resolved_worker_idx}_%p'"
                )
                logging.info(
                    f"Nsight profiling enabled for worker '{worker_name}' "
                    f"(worker_idx={resolved_worker_idx}, matched pattern '{pattern}')"
                )
            else:
                output_name = f"'{worker_name}_{NRL_NSYS_PROFILE_STEP_RANGE}_%p'"
                logging.info(
                    f"Nsight profiling enabled for worker '{worker_name}' (matched pattern '{pattern}')"
                )

            return {
                "nsight": {
                    "t": "cuda,cudnn,cublas,nvtx",
                    "o": output_name,
                    "stop-on-exit": "true",
                    # Capture range is required to control the scope of the profile
                    # Profile will only start/stop when torch.cuda.profiler.start()/stop() is called
                    "capture-range": "cudaProfilerApi",
                    "capture-range-end": "stop",
                    "cuda-graph-trace": "node",
                }
            }

    return {}


def recursive_merge_options(
    default_options: dict[str, Any], extra_options: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge extra options into default options using OmegaConf.

    Args:
        default_options: Default options dictionary (lower precedence)
        extra_options: Extra options provided by the caller (higher precedence)

    Returns:
        Merged options dictionary with extra_options taking precedence over default_options
    """
    # Convert to OmegaConf DictConfig for robust merging
    default_conf = deepcopy(default_options)
    extra_conf = deepcopy(extra_options)

    def recursive_merge_dict(base, incoming):
        """Recursively merge incoming dict into base dict, with incoming taking precedence."""
        if isinstance(incoming, dict):
            for k, v in incoming.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    # Both are dicts, recurse
                    recursive_merge_dict(base[k], v)
                else:
                    # Incoming takes precedence (overwrites base) - handles all cases:
                    # - scalar replacing dict, dict replacing scalar, scalar replacing scalar
                    base[k] = deepcopy(v)

    # Handle special nsight configuration transformation (_nsight -> nsight) early
    # so that extra_options can properly override the transformed default
    # https://github.com/ray-project/ray/blob/3c4a5b65dd492503a707c0c6296820228147189c/python/ray/runtime_env/runtime_env.py#L345
    if "runtime_env" in default_conf and isinstance(default_conf["runtime_env"], dict):
        runtime_env = default_conf["runtime_env"]
        if "_nsight" in runtime_env and "nsight" not in runtime_env:
            runtime_env["nsight"] = runtime_env["_nsight"]
            del runtime_env["_nsight"]

    # Merge in place
    recursive_merge_dict(base=default_conf, incoming=extra_conf)

    return default_conf
