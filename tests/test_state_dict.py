"""Tests for sampler state_dict / load_state_dict resumability."""

from __future__ import annotations

import torch

from mcmc_samplers import (
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    MetropolisAdjustedLangevin,
    MetropolisHastings,
)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _gaussian():
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    prec = torch.linalg.inv(cov)

    def log_prob(x):
        return -0.5 * (x @ prec * x).sum()

    return log_prob


def test_mh_state_dict_round_trip_matches_uninterrupted_run():
    target = _gaussian()
    a = MetropolisHastings(
        target=target, x0=torch.zeros(2), cov=torch.eye(2), generator=_gen(0)
    )
    samples_a, log_probs_a = a(200, show_progress=False)
    state = a.state_dict()

    b = MetropolisHastings(
        target=target, x0=torch.zeros(2), cov=torch.eye(2), generator=_gen(1)
    )
    b.load_state_dict(state)
    samples_b, log_probs_b = b(300, show_progress=False)

    full = MetropolisHastings(
        target=target, x0=torch.zeros(2), cov=torch.eye(2), generator=_gen(0)
    )
    samples_full, log_probs_full = full(500, show_progress=False)

    assert torch.allclose(samples_b, samples_full[200:])
    assert torch.allclose(log_probs_b, log_probs_full[200:])


def test_dram_state_dict_preserves_adaptation():
    target = _gaussian()
    a = DelayedRejectionAdaptiveMetropolis(
        target=target, x0=torch.zeros(2), cov=torch.eye(2), n0=50, generator=_gen(0)
    )
    a(300, show_progress=False)
    state = a.state_dict()

    b = DelayedRejectionAdaptiveMetropolis(
        target=target, x0=torch.zeros(2), cov=torch.eye(2), n0=50, generator=_gen(1)
    )
    b.load_state_dict(state)
    assert b.proposals[0]._iteration == a.proposals[0]._iteration
    assert torch.allclose(b.proposals[0]._mean, a.proposals[0]._mean)
    assert torch.allclose(b.proposals[0].sqrt_cov, a.proposals[0].sqrt_cov)


def test_hmc_state_dict_preserves_gradient():
    target = _gaussian()
    a = HamiltonianMonteCarlo(
        target=target, x0=torch.zeros(2), step_size=0.2, num_steps=3, generator=_gen(0)
    )
    a(100, show_progress=False)
    state = a.state_dict()

    b = HamiltonianMonteCarlo(
        target=target, x0=torch.zeros(2), step_size=0.2, num_steps=3, generator=_gen(1)
    )
    b.load_state_dict(state)
    assert torch.allclose(b._x.point, a._x.point)
    assert torch.allclose(b._x.log_prob, a._x.log_prob)
    assert torch.allclose(b._x.grad, a._x.grad)


def test_mala_state_dict_preserves_gradient():
    target = _gaussian()
    a = MetropolisAdjustedLangevin(
        target=target, x0=torch.zeros(2), step_size=0.3, generator=_gen(0)
    )
    a(100, show_progress=False)
    state = a.state_dict()

    b = MetropolisAdjustedLangevin(
        target=target, x0=torch.zeros(2), step_size=0.3, generator=_gen(1)
    )
    b.load_state_dict(state)
    assert torch.allclose(b._x.point, a._x.point)
