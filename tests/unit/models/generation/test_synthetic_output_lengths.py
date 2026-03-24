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

import asyncio
from copy import deepcopy

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import generate_responses_async
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationOutputSpec
from nemo_rl.models.generation.output_length_samplers import (
    ConstantOutputLengthSampler,
    TruncLognormalFromMeanOutputLengthSampler,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in seq.tolist()) for seq in generated_ids]


class DummySamplingParams(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DummyAsyncPolicyGeneration:
    def __init__(self, outputs):
        self.outputs = outputs
        self.cfg = {"vllm_cfg": {"async_engine": True}}

    async def generate_async(self, generation_input_data, greedy=False):
        del generation_input_data, greedy
        for idx, output in enumerate(self.outputs):
            yield idx, output


def _make_generation_config(**overrides):
    config = {
        "backend": "vllm",
        "max_new_tokens": 32,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "async_engine": True,
            "max_model_len": 128,
            "load_format": "auto",
            "skip_tokenizer_init": False,
        },
    }
    config.update(overrides)
    return config


def _make_worker(cfg):
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker.cfg = cfg
    worker.SamplingParams = DummySamplingParams
    worker._output_length_sampler = None
    worker._initialize_output_length_sampler()
    return worker


def test_configure_generation_config_respects_ignore_eos():
    tokenizer = DummyTokenizer()
    config = configure_generation_config(
        _make_generation_config(ignore_eos=True), tokenizer, is_eval=True
    )

    assert config["ignore_eos"] is True
    assert config["stop_token_ids"] is None
    assert config["_pad_token_id"] == tokenizer.pad_token_id
    assert config["_eos_token_id"] == tokenizer.eos_token_id


def test_configure_generation_config_keeps_default_eos_stop_ids():
    tokenizer = DummyTokenizer()
    config = configure_generation_config(_make_generation_config(), tokenizer, is_eval=True)

    assert config["ignore_eos"] is False
    assert config["stop_token_ids"] == [tokenizer.eos_token_id]


def test_constant_output_length_sampler_returns_constant():
    sampler = ConstantOutputLengthSampler(64)

    assert sampler.sample_one() == 64
    assert sampler.sample(4).tolist() == [64, 64, 64, 64]


def test_trunc_lognormal_sampler_is_seed_deterministic_and_bounded():
    sampler_a = TruncLognormalFromMeanOutputLengthSampler(
        mean_osl=12288, max_osl=49152, p_max=0.98, seed=43
    )
    sampler_b = TruncLognormalFromMeanOutputLengthSampler(
        mean_osl=12288, max_osl=49152, p_max=0.98, seed=43
    )

    sample_a = sampler_a.sample(16)
    sample_b = sampler_b.sample(16)

    assert sample_a.tolist() == sample_b.tolist()
    assert sample_a.min() >= 1
    assert sample_a.max() < 49152
    assert abs(sampler_a.dist["actual_truncated_mean"] - 12288) < 250


def test_sync_dict_output_length_generator_is_rejected():
    cfg = _make_generation_config(
        output_len_or_output_len_generator={
            "type": "trunc_lognormal_from_mean",
            "mean_osl": 128,
            "max_osl": 512,
        }
    )
    cfg["vllm_cfg"]["async_engine"] = False

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker.cfg = cfg

    with pytest.raises(NotImplementedError):
        worker._initialize_output_length_sampler()


def test_build_sampling_params_filters_eos_and_uses_constant_cap():
    cfg = _make_generation_config(
        stop_token_ids=[2, 42],
        ignore_eos=True,
        output_len_or_output_len_generator=64,
    )
    cfg["_eos_token_id"] = 2
    worker = _make_worker(cfg)

    sampling_params = worker._build_sampling_params(greedy=False, stop_strings=None)

    assert sampling_params["max_tokens"] == 32
    assert sampling_params["ignore_eos"] is True
    assert sampling_params["stop_token_ids"] == [42]


def test_resolve_allowed_new_tokens_tracks_clipping():
    cfg = _make_generation_config(output_len_or_output_len_generator=64)
    worker = _make_worker(cfg)

    resolution = worker._resolve_allowed_new_tokens(remaining_ctx=20)

    assert resolution.sampled_output_length == 64
    assert resolution.allowed_new_tokens == 20
    assert resolution.clipped_by_max_new_tokens is True
    assert resolution.clipped_by_context is True
    assert resolution.zero_token_due_to_context is False


def test_generate_responses_async_reports_sampled_output_length_metrics():
    tokenizer = DummyTokenizer()
    batch = BatchedDataDict(
        {
            "message_log": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            "stop_strings": [None, None],
        }
    )
    generation_input_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12], [21, 0]], dtype=torch.long),
            "input_lengths": torch.tensor([2, 1], dtype=torch.long),
        }
    )

    outputs = [
        BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.tensor([[11, 12, 31, 32]], dtype=torch.long),
                "logprobs": torch.zeros((1, 4), dtype=torch.float32),
                "generation_lengths": torch.tensor([2], dtype=torch.long),
                "unpadded_sequence_lengths": torch.tensor([4], dtype=torch.long),
                "sampled_output_lengths": torch.tensor([6], dtype=torch.long),
                "output_length_clipped_by_context": torch.tensor([False], dtype=torch.bool),
                "output_length_clipped_by_max_new_tokens": torch.tensor([True], dtype=torch.bool),
                "zero_token_generations_due_to_context": torch.tensor([False], dtype=torch.bool),
            }
        ),
        BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.tensor([[21, 41, 42]], dtype=torch.long),
                "logprobs": torch.zeros((1, 3), dtype=torch.float32),
                "generation_lengths": torch.tensor([2], dtype=torch.long),
                "unpadded_sequence_lengths": torch.tensor([3], dtype=torch.long),
                "sampled_output_lengths": torch.tensor([9], dtype=torch.long),
                "output_length_clipped_by_context": torch.tensor([True], dtype=torch.bool),
                "output_length_clipped_by_max_new_tokens": torch.tensor([False], dtype=torch.bool),
                "zero_token_generations_due_to_context": torch.tensor([False], dtype=torch.bool),
            }
        ),
    ]

    policy_generation = DummyAsyncPolicyGeneration(outputs)
    _, generated_ids, gen_metrics = asyncio.run(
        generate_responses_async(
            policy_generation,
            generation_input_data,
            batch,
            tokenizer,
            input_lengths=generation_input_data["input_lengths"],
            include_logprobs=True,
            greedy=False,
        )
    )

    assert [seq.tolist() for seq in generated_ids] == [[31, 32], [41, 42]]
    assert gen_metrics["histogram/sampled_output_length"] == [6, 9]
    assert gen_metrics["sampled_output_length/mean"] == pytest.approx(7.5)
    assert gen_metrics["sampled_output_length/max"] == 9
    assert gen_metrics["num_output_length_clipped_by_context"] == 1
    assert gen_metrics["num_output_length_clipped_by_max_new_tokens"] == 1
    assert gen_metrics["num_zero_token_generations_due_to_context"] == 0
