"""
Tests for stochastic (log-normal) execution-time sampling.

The model samples realised execution times W(j,g) ~ LogNormal(mu, sigma) with the
expected value pinned to the empirical stratum mean EXECUTION_TIME_LOOKUP[(k,g)],
so that deterministic mean-based results remain the expected case while the
simulation reflects the empirical within-stratum dispersion.
"""

import math
import random
import statistics

import pytest

from cgrsp.data.execution_times import (
    ExecutionTimeModel,
    GPUType,
    EXECUTION_TIME_LOOKUP,
    EXECUTION_TIME_SIGMA_LN,
    POOLED_SIGMA_LN,
)


def test_mean_is_preserved():
    """The sample mean must converge to the deterministic stratum mean."""
    rng = random.Random(0)
    key = (3, GPUType.RTX_A5000)
    mean = EXECUTION_TIME_LOOKUP[key]
    draws = [ExecutionTimeModel.sample_execution_time(3, GPUType.RTX_A5000, rng)
             for _ in range(200_000)]
    assert abs(statistics.mean(draws) - mean) / mean < 0.01


def test_all_samples_strictly_positive():
    rng = random.Random(1)
    draws = [ExecutionTimeModel.sample_execution_time(1, GPUType.RTX_3090, rng)
             for _ in range(10_000)]
    assert all(d > 0.0 for d in draws)


def test_reproducible_with_same_seed():
    a = [ExecutionTimeModel.sample_execution_time(2, GPUType.RTX_3090, random.Random(42))
         for _ in range(5)]
    b = [ExecutionTimeModel.sample_execution_time(2, GPUType.RTX_3090, random.Random(42))
         for _ in range(5)]
    assert a == b


def test_different_seeds_differ():
    a = ExecutionTimeModel.sample_execution_time(2, GPUType.RTX_3090, random.Random(1))
    b = ExecutionTimeModel.sample_execution_time(2, GPUType.RTX_3090, random.Random(2))
    assert a != b


def test_dispersion_matches_sigma_ln():
    """Empirical log-space std of the draws matches the configured sigma_ln."""
    rng = random.Random(7)
    key = (3, GPUType.RTX_A4000)
    sigma = EXECUTION_TIME_SIGMA_LN[key]
    draws = [ExecutionTimeModel.sample_execution_time(3, GPUType.RTX_A4000, rng)
             for _ in range(200_000)]
    ln_std = statistics.pstdev(math.log(d) for d in draws)
    assert abs(ln_std - sigma) / sigma < 0.03


def test_unknown_stratum_uses_pooled_sigma():
    """A stratum without a fitted sigma falls back to the pooled value."""
    # (1, A4500) has no empirical fit -> pooled fallback
    assert (1, GPUType.RTX_A4500) not in EXECUTION_TIME_SIGMA_LN
    rng = random.Random(3)
    draws = [ExecutionTimeModel.sample_execution_time(1, GPUType.RTX_A4500, rng)
             for _ in range(200_000)]
    ln_std = statistics.pstdev(math.log(d) for d in draws)
    assert abs(ln_std - POOLED_SIGMA_LN) / POOLED_SIGMA_LN < 0.03


def test_zero_sigma_returns_mean_exactly():
    """A degenerate sigma of 0 returns the deterministic mean (sanity guard)."""
    val = ExecutionTimeModel.sample_execution_time(
        1, GPUType.RTX_3090, random.Random(0), sigma_ln=0.0
    )
    assert val == EXECUTION_TIME_LOOKUP[(1, GPUType.RTX_3090)]
