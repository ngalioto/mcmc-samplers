"""Unit tests for the proposal distributions."""

from __future__ import annotations

import pytest
import torch

from mcmc_samplers import (
    AdaptiveCovariance,
    GaussianRandomWalk,
    HamiltonianDynamics,
    HamiltonianSample,
    LangevinProposal,
    PreconditionedCrankNicolson,
    Sample,
    ScaledCovariance,
)
from mcmc_samplers.proposals.gaussian_random_walk import _cholesky_with_jitter


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_grw_propose_shape_and_center():
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    proposal = GaussianRandomWalk(cov=cov)
    center = torch.tensor([3.0, -2.0])
    samples = proposal.propose(
        [Sample(center)], N=5000, generator=_gen(0)
    )
    points = torch.stack([s.point for s in samples])
    assert points.shape == (5000, 2)
    emp_mean = points.mean(dim=0)
    assert torch.allclose(emp_mean, center, atol=0.1)


def test_grw_log_prob_matches_mvn():
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    proposal = GaussianRandomWalk(cov=cov)
    a = Sample(torch.tensor([0.0, 0.0]))
    b = Sample(torch.tensor([1.0, 1.0]))
    mvn = torch.distributions.MultivariateNormal(a.point, cov)
    log_prob = proposal.log_prob([a, b])
    expected = -0.5 * (b.point - a.point) @ torch.linalg.inv(cov) @ (b.point - a.point)
    assert torch.allclose(log_prob, expected, atol=1e-6)


def test_grw_is_symmetric_and_swap():
    cov = torch.tensor([[1.0, 0.3], [0.3, 1.5]])
    proposal = GaussianRandomWalk(cov=cov)
    a = Sample(torch.tensor([0.5, -1.0]))
    b = Sample(torch.tensor([1.0, 2.0]))
    assert proposal.is_symmetric
    assert torch.allclose(proposal.log_prob([a, b]), proposal.log_prob([b, a]))


def test_grw_requires_sqrt_cov_or_cov():
    with pytest.raises(ValueError):
        GaussianRandomWalk()
    with pytest.raises(ValueError):
        GaussianRandomWalk(sqrt_cov=torch.eye(2), cov=torch.eye(2))


def test_cholesky_jitter_fallback_on_singular_matrix():
    # Build a deficient covariance and confirm the jitter path recovers a factor.
    base = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient
    chol = _cholesky_with_jitter(base, eps=1e-8)
    reconstructed = chol @ chol.T
    assert (
        (reconstructed - base).abs().max() < 1e-3
    ), "jitter should give an approximate factor of the input"


def test_adaptive_covariance_matches_empirical_cov_after_n0():
    torch.manual_seed(0)
    init_cov = torch.eye(2)
    proposal = AdaptiveCovariance(cov=init_cov, sd=1.0, n0=50, eps=0.0)

    true_cov = torch.tensor([[1.0, 0.3], [0.3, 2.0]])
    dist = torch.distributions.MultivariateNormal(torch.zeros(2), true_cov)

    points = []
    for _ in range(2000):
        x = dist.sample()
        proposal._adapt(x)
        points.append(x)
    points = torch.stack(points, dim=0)

    n = proposal._iteration
    expected_mean = points.mean(dim=0)
    centered = points - expected_mean
    expected_cov = centered.T @ centered / (n - 1)

    assert torch.allclose(proposal._mean, expected_mean, atol=1e-5)
    assert torch.allclose(proposal._M2 / (n - 1), expected_cov, atol=1e-5)


def test_adaptive_covariance_uses_init_cov_before_n0():
    init_cov = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    proposal = AdaptiveCovariance(cov=init_cov, sd=1.0, n0=100)
    for _ in range(50):
        proposal._adapt(torch.randn(2))
    assert torch.allclose(proposal.cov, init_cov * proposal.sd)


def test_scaled_covariance_view():
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    base = GaussianRandomWalk(cov=cov)
    scaled = ScaledCovariance(base, gamma=0.01)
    assert scaled.is_symmetric
    assert torch.allclose(scaled.sqrt_cov, base.sqrt_cov * (0.01 ** 0.5))


def test_hamiltonian_dynamics_proposal_shape():
    def target(x):
        return -0.5 * x.pow(2).sum()

    dyn = HamiltonianDynamics(target=target, dim=2, step_size=0.1, num_steps=5)
    current = HamiltonianSample(point=torch.tensor([0.0, 0.0]))
    proposed = dyn.propose([current], generator=_gen(0))
    assert isinstance(proposed, HamiltonianSample)
    assert proposed.point.shape == (2,)
    assert proposed.p.shape == (2,)
    assert proposed.log_prob is not None
    assert proposed.grad is not None


def test_hamiltonian_dynamics_mass_matrix_modes():
    def target(x):
        return -0.5 * x.pow(2).sum()

    dyn = HamiltonianDynamics(
        target=target, dim=2, step_size=0.1, num_steps=1,
        mass=torch.tensor([1.0, 4.0]),
    )
    assert dyn._mass.shape == (2,)
    assert torch.allclose(dyn._mass_inv, torch.tensor([1.0, 0.25]))

    dyn_full = HamiltonianDynamics(
        target=target, dim=2, step_size=0.1, num_steps=1,
        mass=torch.tensor([[2.0, 0.5], [0.5, 2.0]]),
    )
    assert dyn_full._mass_inv.shape == (2, 2)


def test_langevin_log_prob_matches_drift_gaussian():
    def target(x):
        return -0.5 * x.pow(2).sum()

    proposal = LangevinProposal(target=target, step_size=0.3)
    a = Sample(torch.tensor([1.0, -0.5]))
    b = Sample(torch.tensor([0.7, -0.2]))
    lp = proposal.log_prob([a, b])
    drift = a.point + 0.5 * (0.3 ** 2) * a.grad  # a.grad populated by log_prob
    expected = -0.5 * ((b.point - drift) / 0.3).pow(2).sum()
    assert torch.allclose(lp, expected, atol=1e-6)


def test_pcn_is_prior_reversible():
    """pCN is not symmetric, but prior(a) q(b|a) == prior(b) q(a|b)."""
    cov = torch.tensor([[1.0, 0.3], [0.3, 2.0]])
    prior = torch.distributions.MultivariateNormal(torch.zeros(2), cov)
    proposal = PreconditionedCrankNicolson(prior_cov=cov, beta=0.3)
    assert not proposal.is_symmetric
    a = Sample(torch.tensor([0.5, -1.0]))
    b = Sample(torch.tensor([-0.2, 0.4]))
    assert not torch.allclose(proposal.log_prob([a, b]), proposal.log_prob([b, a]))
    fwd = prior.log_prob(a.point) + proposal.log_prob([a, b])
    rev = prior.log_prob(b.point) + proposal.log_prob([b, a])
    assert torch.allclose(fwd, rev, atol=1e-5)


def test_pcn_invalid_beta():
    with pytest.raises(ValueError):
        PreconditionedCrankNicolson(prior_cov=torch.eye(2), beta=1.5)
    with pytest.raises(ValueError):
        PreconditionedCrankNicolson(prior_cov=torch.eye(2), beta=0.0)


def test_proposal_state_dict_round_trip():
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    proposal = AdaptiveCovariance(cov=cov, sd=1.2, n0=10)
    for _ in range(20):
        proposal._adapt(torch.randn(2))

    sd = proposal.state_dict()
    proposal2 = AdaptiveCovariance(cov=cov, sd=1.2, n0=10)
    proposal2.load_state_dict(sd)
    assert proposal2._iteration == proposal._iteration
    assert torch.allclose(proposal2._mean, proposal._mean)
    assert torch.allclose(proposal2._M2, proposal._M2)
    assert torch.allclose(proposal2.sqrt_cov, proposal.sqrt_cov)
