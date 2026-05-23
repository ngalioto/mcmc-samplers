"""Targeted tests for AdaptiveCovariance bugs (Cholesky jitter, Welford update)."""

from __future__ import annotations

import pytest
import torch

from mcmc_samplers import AdaptiveCovariance
from mcmc_samplers.proposals.gaussian_random_walk import _cholesky_with_jitter


def test_cholesky_with_jitter_succeeds_on_singular_input():
    """Originally the jitter fallback was no-op due to range(-8, 1, -1) being empty
    and ``10 ^ ii`` being XOR instead of exponentiation. This test would have
    caught both.
    """
    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    chol = _cholesky_with_jitter(singular, eps=1e-6, max_tries=10)
    rebuilt = chol @ chol.T
    assert torch.linalg.matrix_rank(rebuilt) == 2
    assert torch.allclose(rebuilt, singular, atol=1e-2)


def test_adaptive_covariance_no_op_before_n0():
    init = torch.eye(2) * 0.5
    proposal = AdaptiveCovariance(cov=init, sd=1.0, n0=100)
    sqrt_before = proposal.sqrt_cov.clone()
    for _ in range(50):
        proposal._adapt(torch.randn(2))
    assert torch.allclose(proposal.sqrt_cov, sqrt_before)


def test_adaptive_covariance_updates_after_n0():
    init = torch.eye(2)
    proposal = AdaptiveCovariance(cov=init, sd=1.0, n0=20, eps=0.0)
    torch.manual_seed(0)
    true = torch.tensor([[3.0, 0.5], [0.5, 1.0]])
    dist = torch.distributions.MultivariateNormal(torch.zeros(2), true)
    for _ in range(500):
        proposal._adapt(dist.sample())
    expected = proposal._M2 / (proposal._iteration - 1)
    assert torch.allclose(proposal.cov / proposal.sd, expected, atol=1e-5)


def test_adaptive_covariance_does_not_produce_negative_definite():
    """Regression test: the original recursion produced negative-definite
    covariances at iteration 1. The new Welford implementation must keep
    M2 positive semi-definite throughout."""
    init = torch.eye(3)
    proposal = AdaptiveCovariance(cov=init, sd=1.0, n0=10, eps=0.0)
    for _ in range(20):
        proposal._adapt(torch.randn(3) * 0.1)
    eigvals = torch.linalg.eigvalsh(proposal._M2)
    assert (eigvals >= -1e-10).all()
