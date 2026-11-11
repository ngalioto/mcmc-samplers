"""Smoke + convergence tests against simple Gaussian targets.

For each sampler we run a moderate-length chain on a known target and check
that the empirical mean and covariance match the true values within Monte
Carlo tolerance. These tests are necessarily stochastic; we seed the RNG
deterministically to keep them stable.
"""

from __future__ import annotations

import pytest
import torch

from mcmc_samplers import (
    AdaptiveMetropolis,
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    MetropolisAdjustedLangevin,
    MetropolisHastings,
    NUTS,
    ParallelTempering,
    PreconditionedCrankNicolson,
    Proposal,
    Sample,
    Sampler,
    diagnostics,
    run_chains,
)
from mcmc_samplers.proposals.gaussian_random_walk import GaussianRandomWalk
from mcmc_samplers.samplers.delayed_rejection import DelayedRejection


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _check_gaussian_chain(
    samples: torch.Tensor,
    target_mean: torch.Tensor,
    target_cov: torch.Tensor,
    mean_tol: float = 0.2,
    cov_tol: float = 0.4,
    min_ess: float = 30.0,
) -> None:
    """Check moments within 4 Monte Carlo standard errors (ESS-based), never
    tighter than ``mean_tol`` / ``cov_tol``."""
    ess = diagnostics.effective_sample_size(samples).min().clamp(max=samples.shape[0])
    assert ess > min_ess, f"ESS {ess:.0f} too low; chain is not mixing"
    var = target_cov.diagonal()
    mean_se = (var / ess).sqrt()
    cov_se = ((var[:, None] * var[None, :] + target_cov**2) / ess).sqrt()

    emp_mean = samples.mean(dim=0)
    centered = samples - emp_mean
    emp_cov = centered.T @ centered / (samples.shape[0] - 1)
    assert ((emp_mean - target_mean).abs() <= (4 * mean_se).clamp(min=mean_tol)).all(), (
        f"empirical mean {emp_mean} differs from target {target_mean}"
    )
    assert ((emp_cov - target_cov).abs() <= (4 * cov_se).clamp(min=cov_tol)).all(), (
        f"empirical cov {emp_cov} differs from target {target_cov}"
    )


def test_metropolis_hastings_isotropic(correlated_gaussian):
    sampler = MetropolisHastings(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        cov=torch.eye(2),
        generator=_gen(0),
    )
    samples, _ = sampler(4000, burn_in=500, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)
    assert 0.05 < sampler.acceptance_ratio < 0.95


def test_adaptive_metropolis(correlated_gaussian):
    sampler = AdaptiveMetropolis(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        cov=torch.eye(2),
        n0=100,
        generator=_gen(0),
    )
    samples, _ = sampler(4000, burn_in=1000, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)


def test_dram(correlated_gaussian):
    sampler = DelayedRejectionAdaptiveMetropolis(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        cov=torch.eye(2),
        n0=100,
        generator=_gen(0),
    )
    samples, _ = sampler(4000, burn_in=1000, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)


def test_hmc(correlated_gaussian):
    sampler = HamiltonianMonteCarlo(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        step_size=0.2,
        num_steps=5,
        generator=_gen(0),
    )
    samples, _ = sampler(2000, burn_in=500, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)
    assert sampler.acceptance_ratio > 0.5


def test_hmc_with_diagonal_mass(correlated_gaussian):
    sampler = HamiltonianMonteCarlo(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        step_size=0.2,
        num_steps=5,
        mass=torch.tensor([1.0, 2.0]),
        generator=_gen(1),
    )
    samples, _ = sampler(2000, burn_in=500, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)


def test_mala(correlated_gaussian):
    sampler = MetropolisAdjustedLangevin(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        step_size=0.5,
        generator=_gen(0),
    )
    samples, _ = sampler(4000, burn_in=500, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)


def test_nuts(correlated_gaussian):
    sampler = NUTS(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        step_size=0.5,
        warmup=200,
        max_depth=6,
        generator=_gen(0),
    )
    samples, _ = sampler(1500, burn_in=200, show_progress=False)
    _check_gaussian_chain(samples, correlated_gaussian.mean, correlated_gaussian.cov)


def test_parallel_tempering(correlated_gaussian):
    def factory(target_fn, x0):
        return MetropolisHastings(
            target=target_fn,
            x0=x0,
            cov=torch.eye(2),
            generator=_gen(7),
        )

    sampler = ParallelTempering(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        sampler_factory=factory,
        betas=(1.0, 0.5, 0.25),
        generator=_gen(0),
    )
    samples, _ = sampler(2000, burn_in=500, show_progress=False)
    _check_gaussian_chain(
        samples, correlated_gaussian.mean, correlated_gaussian.cov, mean_tol=0.3
    )


def test_pcn_recovers_prior_when_likelihood_is_flat():
    """With a flat (zero) likelihood, pCN should sample the prior."""
    from mcmc_samplers import PreconditionedCrankNicolsonSampler

    prior_cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

    def likelihood(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=x.dtype, device=x.device)

    sampler = PreconditionedCrankNicolsonSampler(
        likelihood=likelihood,
        x0=torch.zeros(2),
        prior_cov=prior_cov,
        beta=0.4,
        generator=_gen(0),
    )
    samples, _ = sampler(6000, burn_in=1000, show_progress=False)
    _check_gaussian_chain(samples, torch.zeros(2), prior_cov, mean_tol=0.25)


def test_run_chains_multi_chain(correlated_gaussian):
    def factory(chain_idx: int) -> Sampler:
        return MetropolisHastings(
            target=correlated_gaussian,
            x0=torch.tensor([float(chain_idx), -float(chain_idx)]),
            cov=torch.eye(2),
            generator=_gen(chain_idx + 1),
        )

    chains, log_probs = run_chains(
        factory, num_chains=3, N=1500, burn_in=300, show_progress=False
    )
    assert chains.shape == (3, 1200, 2)
    assert log_probs.shape == (3, 1200)
    for c in range(3):
        _check_gaussian_chain(
            chains[c], correlated_gaussian.mean, correlated_gaussian.cov, mean_tol=0.3
        )


def test_hmc_rejects_divergent_trajectories():
    """A step size far too large sends the trajectory to inf/NaN. That must be
    rejected, not passed to the target (which may raise on invalid input)."""
    dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    sampler = HamiltonianMonteCarlo(
        target=dist.log_prob,
        x0=torch.zeros(2),
        step_size=10.0,
        num_steps=500,
        generator=_gen(0),
    )
    samples, log_probs = sampler(20, show_progress=False)
    assert torch.isfinite(samples).all() and torch.isfinite(log_probs).all()
    assert sampler.acceptance_ratio < 0.5
