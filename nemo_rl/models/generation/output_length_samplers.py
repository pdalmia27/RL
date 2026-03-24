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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol, cast

import numpy as np

from nemo_rl.models.generation.interfaces import (
    OutputLengthGeneratorConfig,
    TruncLognormalFromMeanOutputLengthGeneratorConfig,
)

try:
    from scipy.optimize import brentq as _scipy_brentq
    from scipy.stats import norm as _scipy_norm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _scipy_brentq = None
    _scipy_norm = None

_NORMAL_DIST = NormalDist()


def _validate_probability(p: float, name: str = "p_max") -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {p}")
    return float(p)


def _normal_cdf(x: float) -> float:
    if _scipy_norm is not None:
        return float(_scipy_norm.cdf(x))
    return _NORMAL_DIST.cdf(x)


def _normal_ppf(p: float) -> float:
    p = _validate_probability(p)
    if _scipy_norm is not None:
        return float(_scipy_norm.ppf(p))
    return _NORMAL_DIST.inv_cdf(p)


def _truncated_lognormal_moment(
    mu: float,
    sigma: float,
    upper_bound: float,
    k: int,
) -> float:
    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return 1.0

    ln_b = math.log(upper_bound)
    z_b = (ln_b - mu) / sigma
    phi_z_b = _normal_cdf(z_b)
    if phi_z_b < 1e-15:
        return (upper_bound**k) / 2.0

    z_b_shifted = (ln_b - mu - k * (sigma**2)) / sigma
    untruncated = math.exp(k * mu + 0.5 * (k**2) * (sigma**2))
    return untruncated * _normal_cdf(z_b_shifted) / phi_z_b


def truncated_lognormal_mean(mu: float, sigma: float, upper_bound: float) -> float:
    """Compute E[X | X <= upper_bound] for a log-normal distribution."""
    return _truncated_lognormal_moment(mu, sigma, upper_bound, 1)


def lognormal_params_from_mean_and_max(
    s_mean: float,
    s_max: float,
    p_max: float = 0.99,
) -> tuple[float, float]:
    if s_mean <= 0 or s_max <= 0:
        raise ValueError("s_mean and s_max must be positive.")
    if s_max <= s_mean:
        raise ValueError(
            f"s_max ({s_max}) must be greater than s_mean ({s_mean})."
        )

    z_p = _normal_ppf(_validate_probability(p_max))
    c = math.log(s_max / s_mean)
    discriminant = z_p**2 - 2 * c

    if discriminant < 0:
        raise ValueError(
            f"Cannot satisfy Mean={s_mean}, Max={s_max} at p={p_max} with LogNormal shape constraints."
        )

    sigma = z_p - math.sqrt(discriminant)
    if sigma <= 0:
        sigma = 1e-6

    mu = math.log(s_max) - sigma * z_p
    return mu, sigma


def _find_inflated_mean_for_truncated_target(
    target_mean: float,
    max_osl: float,
    p_max: float = 0.99,
    tol: float = 1.0,
    max_iter: int = 50,
) -> tuple[float, float]:
    """Mirror DLSim's inversion of truncated-mean targets into base lognormal params."""

    lo = float(target_mean)
    hi = float(target_mean * 3.0)

    def objective(inflated_mean: float) -> float:
        try:
            mu, sigma = lognormal_params_from_mean_and_max(
                inflated_mean, max_osl, p_max
            )
            trunc_mean = truncated_lognormal_mean(mu, sigma, max_osl)
            return trunc_mean - target_mean
        except ValueError:
            return float("inf")

    lo_val = objective(lo)
    if lo_val > 0:
        return lognormal_params_from_mean_and_max(lo, max_osl, p_max)

    hi_val = objective(hi)
    expand_steps = 0
    while hi_val < 0 and expand_steps < max_iter:
        hi *= 2.0
        hi_val = objective(hi)
        expand_steps += 1

    if hi_val < 0:
        return lognormal_params_from_mean_and_max(hi, max_osl, p_max)

    if _scipy_brentq is not None:
        root = _scipy_brentq(objective, lo, hi, xtol=float(tol), maxiter=int(max_iter))
    else:  # pragma: no cover - exercised only when scipy is absent
        left, right = lo, hi
        root = right
        for _ in range(max_iter):
            root = 0.5 * (left + right)
            root_val = objective(root)
            if abs(root_val) <= tol or abs(right - left) <= tol:
                break
            if root_val > 0:
                right = root
            else:
                left = root

    return lognormal_params_from_mean_and_max(float(root), max_osl, p_max)


def get_or_create_trunc_lognormal_dist_from_mean(
    mean_osl: int,
    max_osl: int,
    *,
    p_max: float = 0.98,
    version: int = 5,
) -> dict[str, float]:
    mu, sigma = _find_inflated_mean_for_truncated_target(
        float(mean_osl), float(max_osl), p_max
    )

    median_osl = math.exp(mu)
    actual_trunc_mean = truncated_lognormal_mean(mu, sigma, float(max_osl))

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "median_osl": float(median_osl),
        "input_mean_osl": float(mean_osl),
        "actual_truncated_mean": float(actual_trunc_mean),
        "max_osl": float(max_osl),
        "p_max": float(p_max),
        "version": float(version),
    }


def sample_truncated_lognormal_int(
    *,
    mu: float,
    sigma: float,
    size: int,
    max_osl: int,
    seed: int = 43,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng(seed)

    lo = 1.0
    hi = float(max_osl)
    if hi <= lo:
        raise ValueError(
            f"Invalid truncation bounds for lognormal sampling: lo={lo}, hi={hi} (max_osl={max_osl})"
        )

    out = np.empty(int(size), dtype=np.int64)
    filled = 0
    empty_draws = 0
    max_empty_draws = 1000
    while filled < size:
        batch = (size - filled) * 2
        draw = rng.lognormal(mean=float(mu), sigma=float(sigma), size=int(batch))
        draw = draw[(draw >= lo) & (draw < hi)]
        if draw.size == 0:
            empty_draws += 1
            if empty_draws >= max_empty_draws:
                raise RuntimeError(
                    "Rejection sampling produced no values within bounds after "
                    f"{max_empty_draws} attempts (mu={mu}, sigma={sigma}, lo={lo}, hi={hi}, "
                    f"size={size}, filled={filled})."
                )
            continue
        empty_draws = 0
        take = min(int(draw.size), int(size - filled))
        out[filled : filled + take] = np.floor(draw[:take]).astype(np.int64)
        filled += take

    out[out < 1] = 1
    if max_osl > 1:
        out[out >= max_osl] = max_osl - 1
    return out


class OutputLengthSampler(Protocol):
    def sample(self, size: int = 1) -> np.ndarray: ...

    def sample_one(self) -> int: ...


@dataclass
class ConstantOutputLengthSampler:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError(
                f"Constant output length must be positive, got {self.value}"
            )

    def sample(self, size: int = 1) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=np.int64)
        return np.full(size, self.value, dtype=np.int64)

    def sample_one(self) -> int:
        return self.value


@dataclass
class TruncLognormalFromMeanOutputLengthSampler:
    mean_osl: int
    max_osl: int
    p_max: float = 0.98
    seed: int = 43
    dist: dict[str, float] = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.mean_osl <= 0:
            raise ValueError(
                f"mean_osl must be positive for synthetic output-length sampling, got {self.mean_osl}"
            )
        if self.max_osl <= self.mean_osl:
            raise ValueError(
                f"max_osl must be greater than mean_osl, got mean_osl={self.mean_osl} max_osl={self.max_osl}"
            )
        self.dist = get_or_create_trunc_lognormal_dist_from_mean(
            self.mean_osl,
            self.max_osl,
            p_max=self.p_max,
        )
        self.rng = np.random.default_rng(self.seed)

    def sample(self, size: int = 1) -> np.ndarray:
        return sample_truncated_lognormal_int(
            mu=self.dist["mu"],
            sigma=self.dist["sigma"],
            size=size,
            max_osl=self.max_osl,
            rng=self.rng,
        )

    def sample_one(self) -> int:
        return int(self.sample(1)[0])


def build_output_length_sampler(
    config: OutputLengthGeneratorConfig | int | None,
) -> OutputLengthSampler | None:
    if config is None:
        return None
    if isinstance(config, int):
        return ConstantOutputLengthSampler(config)
    if not isinstance(config, Mapping):
        raise TypeError(
            "output_len_or_output_len_generator must be an int, a mapping, or None"
        )

    generator_type = config.get("type")
    if generator_type != "trunc_lognormal_from_mean":
        raise ValueError(
            f"Unsupported output length generator type: {generator_type!r}"
        )

    typed_config = cast(TruncLognormalFromMeanOutputLengthGeneratorConfig, config)
    return TruncLognormalFromMeanOutputLengthSampler(
        mean_osl=int(typed_config["mean_osl"]),
        max_osl=int(typed_config["max_osl"]),
        p_max=float(typed_config.get("p_max", 0.98)),
        seed=int(typed_config.get("seed", 43)),
    )
