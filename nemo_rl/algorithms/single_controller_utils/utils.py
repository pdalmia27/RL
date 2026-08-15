# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {"probs_ratio_max", "probs_ratio_clamped_max"}
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)


@dataclass
class ImportanceSamplingDiagnosticsAccumulator:
    """Accumulate raw actor/behavior log-ratio diagnostics for one train step."""

    use_importance_sampling_correction: bool
    sequence_level_importance_ratios: bool
    truncated_importance_sampling_ratio: float | None
    truncated_importance_sampling_ratio_min: float | None
    truncated_importance_sampling_type: str | None
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _token_log_ratios_by_lag: dict[int, list[torch.Tensor]] = field(
        default_factory=dict, init=False
    )

    @staticmethod
    def _quantiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[float]:
        if values.numel() == 0:
            return [0.0] * len(qs)
        quantiles = torch.quantile(values.float(), torch.tensor(qs))
        return [float(value) for value in quantiles]

    def _post_tis_weights(
        self, log_ratios: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        upper = self.truncated_importance_sampling_ratio
        lower = self.truncated_importance_sampling_ratio_min
        tis_type = self.truncated_importance_sampling_type
        raw_token_weights = torch.nan_to_num(
            torch.exp(log_ratios), nan=0.0, posinf=0.0, neginf=0.0
        )

        def outside_bounds(value: torch.Tensor) -> bool:
            return bool(
                (upper is not None and value > math.log(upper))
                or (lower is not None and lower > 0 and value < math.log(lower))
            )

        if self.sequence_level_importance_ratios:
            sequence_log_ratio = log_ratios.sum()
            sequence_weight = torch.nan_to_num(
                torch.exp(sequence_log_ratio), nan=0.0, posinf=0.0, neginf=0.0
            )
            raw_weights = sequence_weight.expand_as(raw_token_weights)
            oob = torch.full_like(
                log_ratios,
                outside_bounds(sequence_log_ratio),
                dtype=torch.bool,
            )
        elif tis_type == "seq-mask-tis":
            sequence_mean_log_ratio = log_ratios.mean()
            raw_weights = raw_token_weights
            oob = torch.full_like(
                log_ratios,
                outside_bounds(sequence_mean_log_ratio),
                dtype=torch.bool,
            )
        else:
            raw_weights = raw_token_weights
            oob = torch.zeros_like(log_ratios, dtype=torch.bool)
            if upper is not None:
                oob |= log_ratios > math.log(upper)
            if lower is not None and lower > 0:
                oob |= log_ratios < math.log(lower)

        if tis_type is None:
            return raw_weights, oob
        if tis_type == "tis":
            assert upper is not None
            return raw_weights.clamp(min=lower or 0.0, max=upper), oob
        if tis_type == "icepop":
            return torch.where(oob, torch.zeros_like(raw_weights), raw_weights), oob
        if tis_type == "seq-mask-tis":
            if bool(oob.any()):
                return torch.zeros_like(raw_weights), torch.ones_like(oob)
            return raw_weights, oob
        raise ValueError(f"unsupported truncated importance sampling type: {tis_type}")

    def record(
        self,
        *,
        step: int,
        trainer_version: int,
        sample_ids: list[str],
        rollout_weight_versions: list[int],
        sequence_lengths: list[int] | None,
        prev_logprobs: torch.Tensor,
        generation_logprobs: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        """Record one selected sampler tranche before its optimizer update."""
        log_ratios = (prev_logprobs[:, 1:] - generation_logprobs[:, 1:]).float()
        valid_mask = token_mask[:, 1:].bool() & sample_mask.bool().unsqueeze(-1)
        next_token_advantages = advantages[:, 1:].float()
        batch_size = log_ratios.shape[0]
        if not (
            len(sample_ids) == len(rollout_weight_versions) == batch_size
            and rewards.numel() == batch_size
            and (sequence_lengths is None or len(sequence_lengths) == batch_size)
            and all(
                tensor.shape[0] == batch_size
                for tensor in (
                    generation_logprobs,
                    token_mask,
                    sample_mask,
                    advantages,
                )
            )
        ):
            raise ValueError(
                "importance-sampling diagnostic batch metadata is misaligned"
            )

        for i, sample_id in enumerate(sample_ids):
            response_token_count = int(valid_mask[i].sum())
            finite_mask = valid_mask[i] & torch.isfinite(log_ratios[i])
            token_log_ratios = log_ratios[i][finite_mask].detach().cpu()
            token_advantages = next_token_advantages[i][finite_mask].detach().cpu()
            finite_token_count = int(token_log_ratios.numel())
            rollout_version = int(rollout_weight_versions[i])
            lag = int(trainer_version - rollout_version)

            if finite_token_count > 0:
                post_tis_weights, oob_mask = self._post_tis_weights(token_log_ratios)
                objective_weights = (
                    post_tis_weights
                    if self.use_importance_sampling_correction
                    else torch.ones_like(post_tis_weights)
                )
                signal_proxy = (token_advantages.abs() * objective_weights).mean()
                self._token_log_ratios_by_lag.setdefault(lag, []).append(
                    token_log_ratios
                )
                sequence_sum_log_ratio = float(token_log_ratios.sum())
                sequence_mean_log_ratio = float(token_log_ratios.mean())
                token_abs_log_ratio_mean = float(token_log_ratios.abs().mean())
                post_tis_ratio_mean = float(post_tis_weights.mean())
                tis_oob_fraction = float(oob_mask.float().mean())
                nonzero_advantage_fraction = float(
                    token_advantages.ne(0).float().mean()
                )
                objective_signal_proxy_mean = float(signal_proxy)
                token_quantiles = self._quantiles(
                    token_log_ratios, (0.05, 0.50, 0.95, 0.99)
                )
            else:
                sequence_sum_log_ratio = 0.0
                sequence_mean_log_ratio = 0.0
                token_abs_log_ratio_mean = 0.0
                post_tis_ratio_mean = 0.0
                tis_oob_fraction = 0.0
                nonzero_advantage_fraction = 0.0
                objective_signal_proxy_mean = 0.0
                token_quantiles = [0.0, 0.0, 0.0, 0.0]

            self._rows.append(
                {
                    "step": int(step),
                    "sample_id": sample_id,
                    "prompt_group_id": sample_id.rsplit("_g", 1)[0],
                    "rollout_weight_version": rollout_version,
                    "trainer_weight_version": int(trainer_version),
                    "observed_lag": lag,
                    "total_sequence_length": (
                        int(sequence_lengths[i])
                        if sequence_lengths is not None
                        else None
                    ),
                    "response_token_count": response_token_count,
                    "finite_log_ratio_token_count": finite_token_count,
                    "nonfinite_log_ratio_token_count": (
                        response_token_count - finite_token_count
                    ),
                    "reward": float(rewards.flatten()[i]),
                    "raw_sequence_sum_log_ratio": sequence_sum_log_ratio,
                    "raw_sequence_mean_log_ratio": sequence_mean_log_ratio,
                    "raw_token_abs_log_ratio_mean": token_abs_log_ratio_mean,
                    "raw_token_log_ratio_p05": token_quantiles[0],
                    "raw_token_log_ratio_p50": token_quantiles[1],
                    "raw_token_log_ratio_p95": token_quantiles[2],
                    "raw_token_log_ratio_p99": token_quantiles[3],
                    "post_tis_ratio_mean": post_tis_ratio_mean,
                    "tis_oob_fraction": tis_oob_fraction,
                    "nonzero_advantage_fraction": nonzero_advantage_fraction,
                    "objective_signal_proxy_mean": objective_signal_proxy_mean,
                }
            )

    @staticmethod
    def _weighted_row_mean(rows: list[dict[str, Any]], field_name: str) -> float:
        total_tokens = sum(int(row["finite_log_ratio_token_count"]) for row in rows)
        if total_tokens == 0:
            return 0.0
        return (
            sum(
                float(row[field_name]) * int(row["finite_log_ratio_token_count"])
                for row in rows
            )
            / total_tokens
        )

    def _summarize_bucket(
        self,
        *,
        label: str,
        rows: list[dict[str, Any]],
        token_log_ratios: torch.Tensor,
    ) -> dict[str, float]:
        prefix = f"importance_sampling/{label}"
        sequence_means = torch.tensor(
            [float(row["raw_sequence_mean_log_ratio"]) for row in rows]
        )
        sequence_sums = torch.tensor(
            [float(row["raw_sequence_sum_log_ratio"]) for row in rows]
        )
        response_tokens = torch.tensor(
            [float(row["response_token_count"]) for row in rows]
        )
        rewards = torch.tensor([float(row["reward"]) for row in rows])
        token_quantiles = self._quantiles(token_log_ratios, (0.05, 0.50, 0.95, 0.99))
        sequence_quantiles = self._quantiles(sequence_means, (0.50, 0.95))
        sequence_sum_quantiles = self._quantiles(sequence_sums, (0.50, 0.95))
        return {
            f"{prefix}/num_sequences": float(len(rows)),
            f"{prefix}/num_tokens": float(token_log_ratios.numel()),
            f"{prefix}/num_nonfinite_tokens": float(
                sum(int(row["nonfinite_log_ratio_token_count"]) for row in rows)
            ),
            f"{prefix}/raw_token_log_ratio_mean": (
                float(token_log_ratios.mean()) if token_log_ratios.numel() else 0.0
            ),
            f"{prefix}/raw_token_abs_log_ratio_mean": (
                float(token_log_ratios.abs().mean())
                if token_log_ratios.numel()
                else 0.0
            ),
            f"{prefix}/raw_token_log_ratio_p05": token_quantiles[0],
            f"{prefix}/raw_token_log_ratio_p50": token_quantiles[1],
            f"{prefix}/raw_token_log_ratio_p95": token_quantiles[2],
            f"{prefix}/raw_token_log_ratio_p99": token_quantiles[3],
            f"{prefix}/raw_sequence_mean_log_ratio_p50": sequence_quantiles[0],
            f"{prefix}/raw_sequence_mean_log_ratio_p95": sequence_quantiles[1],
            f"{prefix}/raw_sequence_sum_log_ratio_p50": sequence_sum_quantiles[0],
            f"{prefix}/raw_sequence_sum_log_ratio_p95": sequence_sum_quantiles[1],
            f"{prefix}/post_tis_ratio_mean": self._weighted_row_mean(
                rows, "post_tis_ratio_mean"
            ),
            f"{prefix}/tis_oob_fraction": self._weighted_row_mean(
                rows, "tis_oob_fraction"
            ),
            f"{prefix}/nonzero_advantage_fraction": self._weighted_row_mean(
                rows, "nonzero_advantage_fraction"
            ),
            f"{prefix}/objective_signal_proxy_mean": self._weighted_row_mean(
                rows, "objective_signal_proxy_mean"
            ),
            f"{prefix}/reward_mean": float(rewards.mean()) if rewards.numel() else 0.0,
            f"{prefix}/response_tokens_mean": (
                float(response_tokens.mean()) if response_tokens.numel() else 0.0
            ),
        }

    def flush(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return step metrics and rows, then reset the accumulator."""
        if not self._rows:
            return {}, []

        metrics: dict[str, float] = {}
        all_tokens = [
            values
            for lag_values in self._token_log_ratios_by_lag.values()
            for values in lag_values
        ]
        metrics.update(
            self._summarize_bucket(
                label="all",
                rows=self._rows,
                token_log_ratios=(
                    torch.cat(all_tokens) if all_tokens else torch.empty(0)
                ),
            )
        )
        for lag in sorted({int(row["observed_lag"]) for row in self._rows}):
            lag_rows = [row for row in self._rows if row["observed_lag"] == lag]
            lag_tokens = self._token_log_ratios_by_lag.get(lag, [])
            metrics.update(
                self._summarize_bucket(
                    label=f"lag_{lag}",
                    rows=lag_rows,
                    token_log_ratios=(
                        torch.cat(lag_tokens) if lag_tokens else torch.empty(0)
                    ),
                )
            )

        rows = self._rows
        self._rows = []
        self._token_log_ratios_by_lag = {}
        return metrics, rows


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def reduce_advantage_pump_metrics(
    rewards: list[torch.Tensor],
    masked_advantages: list[torch.Tensor],
    sequence_lengths: list[int],
) -> dict[str, float]:
    """Reduce per-step accumulators from _advantage_stage into step scalars.

    Args:
        rewards: One tensor per advantage_stage call; each row a sample reward.
        masked_advantages: Token-masked advantages, one tensor per call.
        sequence_lengths: All input_lengths trained on this step.

    Returns:
        Dict with reward, advantages/{mean,max,min}, total_num_tokens.
    """
    out: dict[str, float] = {}
    if rewards:
        out["reward"] = float(torch.cat([r.flatten() for r in rewards]).mean())
    if masked_advantages:
        cat = torch.cat([a.flatten() for a in masked_advantages])
        if cat.numel() > 0:
            out["advantages/mean"] = float(cat.mean())
            out["advantages/max"] = float(cat.max())
            out["advantages/min"] = float(cat.min())
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    return out


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
