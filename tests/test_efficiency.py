"""Targeted tests for efficiency-related invariants.

These tests don't measure wall-clock performance -- they pin down the
observable invariants that make the optimisations safe:

* MALA's :meth:`propose` populates ``log_prob`` and ``grad`` eagerly on the
  returned sample, so :class:`DelayedRejection._log_alpha` never has to call
  the target a second time just to fill in ``log_prob``.
* The symmetric Welford update keeps ``_M2`` exactly symmetric step by step,
  and is numerically equivalent to the original Welford recursion.
* ``cache_identity`` and ``adapt_interval`` behave as documented and the
  proposal still converges to the empirical covariance.
* HMC's accepted state adopts the proposed ``HamiltonianSample`` without
  cloning (i.e. they share underlying storage).
"""

from __future__ import annotations

import pytest
import torch

from mcmc_samplers import (
    AdaptiveCovariance,
    AdaptiveMetropolis,
    HamiltonianMonteCarlo,
    HamiltonianSample,
    LangevinProposal,
    MetropolisAdjustedLangevin,
    NUTS,
    Sample,
)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# -----------------------------------------------------------------------------
# MALA eager-gradient invariants
# -----------------------------------------------------------------------------


def test_mala_propose_returns_sample_with_log_prob_and_grad():
    def target(x):
        return -0.5 * x.pow(2).sum()

    proposal = LangevinProposal(target=target, step_size=0.3)
    current = Sample(point=torch.tensor([1.0, -0.5]))
    proposed = proposal.propose([current], generator=_gen(0))

    assert isinstance(proposed, Sample)
    assert proposed.log_prob is not None, "eager log_prob expected on proposed sample"
    assert proposed.grad is not None, "eager grad expected on proposed sample"

    # Verify the eager cache holds the correct values.
    expected = target(proposed.point)
    assert torch.allclose(proposed.log_prob, expected.squeeze(), atol=1e-6)
    expected_grad = -proposed.point
    assert torch.allclose(proposed.grad, expected_grad, atol=1e-6)


def test_mala_init_seeds_grad_on_x0():
    """Constructing MALA should populate ``_x.grad`` so iteration 1's propose
    doesn't waste a forward-only target eval."""

    def target(x):
        return -0.5 * x.pow(2).sum()

    sampler = MetropolisAdjustedLangevin(
        target=target,
        x0=torch.tensor([1.5, -0.7]),
        step_size=0.3,
        generator=_gen(0),
    )
    assert sampler._x.log_prob is not None
    assert sampler._x.grad is not None
    assert torch.allclose(sampler._x.grad, -sampler._x.point, atol=1e-6)


def test_mala_step_uses_exactly_one_target_eval_per_iter():
    """After init, each MALA step should call ``target`` exactly once (at the
    newly proposed point). The eager-grad fix removes the duplicate forward
    eval in ``DelayedRejection._log_alpha``.

    The MALA x.setter override also consolidates the initial state setup into
    a single forward+backward pass, so initialisation costs exactly one
    target call too.
    """

    call_counter = {"n": 0}

    def target(x):
        call_counter["n"] += 1
        return -0.5 * x.pow(2).sum()

    sampler = MetropolisAdjustedLangevin(
        target=target,
        x0=torch.tensor([0.3, -0.1]),
        step_size=0.4,
        generator=_gen(1),
    )
    # Exactly one ``log_prob_and_grad`` call inside the MALA setter; this
    # populates both ``log_prob`` and ``grad`` in one autograd pass.
    assert call_counter["n"] == 1, (
        f"expected 1 target eval during MALA init, got {call_counter['n']}"
    )

    call_counter["n"] = 0
    sampler(N=10, burn_in=0, thin=1, show_progress=False)
    # Exactly one target eval per iteration in steady state (eager grad
    # on the proposed sample; everything else is cached).
    assert call_counter["n"] == 10, (
        f"expected 10 target evals for 10 MALA steps, got {call_counter['n']}"
    )


# -----------------------------------------------------------------------------
# Adaptive covariance: symmetric Welford
# -----------------------------------------------------------------------------


def test_adaptive_covariance_m2_is_exactly_symmetric():
    """The symmetric Welford update guarantees that ``_M2`` stays bit-exactly
    symmetric (no floating-point asymmetry). This was not true for the old
    asymmetric ``outer(delta, delta2)`` update."""

    torch.manual_seed(123)
    proposal = AdaptiveCovariance(cov=torch.eye(3), sd=1.0, n0=5, eps=0.0)
    for _ in range(200):
        proposal._adapt(torch.randn(3) * 2.7 - 0.4)
        # Bit-exact symmetry, not just allclose.
        assert torch.equal(proposal._M2, proposal._M2.T), (
            "symmetric Welford must produce a bit-symmetric _M2 every step"
        )


def test_adaptive_covariance_welford_matches_offline_covariance():
    """End-to-end: the new Welford accumulator must match the offline
    sample covariance to numerical precision."""

    torch.manual_seed(7)
    cov = torch.tensor([[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 1.4]])
    dist = torch.distributions.MultivariateNormal(torch.zeros(3), cov)
    points = dist.sample((500,))

    proposal = AdaptiveCovariance(cov=torch.eye(3), sd=1.0, n0=1000, eps=0.0)
    for x in points:
        proposal._adapt(x)

    n = proposal._iteration
    expected_mean = points.mean(dim=0)
    centered = points - expected_mean
    expected_cov = centered.T @ centered / (n - 1)

    assert torch.allclose(proposal._mean, expected_mean, atol=1e-10)
    assert torch.allclose(proposal._M2 / (n - 1), expected_cov, atol=1e-10)


# -----------------------------------------------------------------------------
# Adaptive covariance: adapt_interval
# -----------------------------------------------------------------------------


def test_adapt_interval_default_recomputes_every_step():
    """Default ``adapt_interval=1`` must recompute ``sqrt_cov`` on every
    post-``n0`` step."""

    proposal = AdaptiveCovariance(
        cov=torch.eye(2), sd=1.0, n0=3, eps=1e-8, adapt_interval=1
    )
    torch.manual_seed(0)
    cov_after = []
    for _ in range(8):
        proposal._adapt(torch.randn(2))
        cov_after.append(proposal.sqrt_cov.clone())
    # First three iterations (n <= n0=3) still use the fixed cov_0 sqrt.
    # Iterations 4..8 should all differ from each other as M2 keeps changing.
    for a, b in zip(cov_after[3:], cov_after[4:]):
        assert not torch.allclose(a, b)


def test_adapt_interval_skips_intermediate_recompute():
    """``adapt_interval=5`` should leave ``sqrt_cov`` unchanged between
    recompute boundaries even though ``_mean``/``_M2`` keep updating."""

    proposal = AdaptiveCovariance(
        cov=torch.eye(2), sd=1.0, n0=2, eps=1e-8, adapt_interval=5
    )
    torch.manual_seed(0)
    # Warm up past n0.
    for _ in range(3):
        proposal._adapt(torch.randn(2))
    # n now == 3. Next recompute fires at n = n0 + adapt_interval = 2 + 5 = 7.
    proposal._adapt(torch.randn(2))  # n=4
    sqrt_after_first_post_n0 = proposal.sqrt_cov.clone()
    for _ in range(2):  # n=5, n=6
        proposal._adapt(torch.randn(2))
        assert torch.equal(proposal.sqrt_cov, sqrt_after_first_post_n0), (
            "sqrt_cov should be unchanged between adapt_interval boundaries"
        )
    proposal._adapt(torch.randn(2))  # n=7, boundary
    assert not torch.equal(proposal.sqrt_cov, sqrt_after_first_post_n0)


def test_adapt_interval_invalid():
    with pytest.raises(ValueError):
        AdaptiveCovariance(cov=torch.eye(2), adapt_interval=0)
    with pytest.raises(ValueError):
        AdaptiveCovariance(cov=torch.eye(2), adapt_interval=-3)


# -----------------------------------------------------------------------------
# Adaptive covariance: identity cache
# -----------------------------------------------------------------------------


def test_identity_cache_default_on():
    proposal = AdaptiveCovariance(cov=torch.eye(2))
    assert proposal.cache_identity is True
    assert proposal._identity is not None
    assert torch.equal(proposal._identity, torch.eye(2))


def test_identity_cache_can_be_disabled():
    proposal = AdaptiveCovariance(cov=torch.eye(2), cache_identity=False)
    assert proposal.cache_identity is False
    assert proposal._identity is None
    # ``cov`` should still work via a fresh allocation each call.
    proposal._iteration = 5  # force the empirical-cov branch
    proposal.n0 = 2
    proposal._M2 = torch.eye(2)
    cov_first = proposal.cov
    cov_second = proposal.cov
    # Returns equivalent results without caching.
    assert torch.allclose(cov_first, cov_second)


def test_identity_cache_survives_state_dict_round_trip():
    proposal = AdaptiveCovariance(
        cov=torch.eye(2), cache_identity=False, adapt_interval=4
    )
    sd = proposal.state_dict()
    proposal2 = AdaptiveCovariance(cov=torch.eye(2))
    proposal2.load_state_dict(sd)
    assert proposal2.cache_identity is False
    assert proposal2._identity is None
    assert proposal2.adapt_interval == 4


def test_adaptive_metropolis_propagates_options(correlated_gaussian):
    sampler = AdaptiveMetropolis(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        cov=torch.eye(2),
        n0=50,
        cache_identity=False,
        adapt_interval=10,
        generator=_gen(0),
    )
    proposal = sampler.proposals[0]
    assert proposal.cache_identity is False
    assert proposal.adapt_interval == 10
    # Chain still runs end-to-end.
    samples, _ = sampler(400, burn_in=100, show_progress=False)
    assert samples.shape == (300, 2)


# -----------------------------------------------------------------------------
# HMC/NUTS no-clone hot path
# -----------------------------------------------------------------------------


def test_hmc_setter_adopts_hamiltonian_sample_without_cloning():
    """The fast-path HMC ``x.setter`` should adopt a HamiltonianSample by
    reference (no defensive clone) -- this saves three tensor copies per
    accepted step."""

    def target(x):
        return -0.5 * x.pow(2).sum()

    sampler = HamiltonianMonteCarlo(
        target=target,
        x0=torch.zeros(2),
        step_size=0.1,
        num_steps=1,
        generator=_gen(0),
    )
    point_tensor = torch.tensor([0.4, -0.2])
    grad_tensor = torch.tensor([-0.4, 0.2])
    fresh = HamiltonianSample(
        point=point_tensor,
        log_prob=torch.tensor(-0.1),
        grad=grad_tensor,
    )
    point_ptr_before = point_tensor.data_ptr()
    grad_ptr_before = grad_tensor.data_ptr()

    sampler.x = fresh

    assert sampler._x is fresh, "HMC setter should adopt the HamiltonianSample"
    # Underlying tensor storage must not have been cloned.
    assert sampler._x.point.data_ptr() == point_ptr_before
    assert sampler._x.grad.data_ptr() == grad_ptr_before


def test_nuts_setter_adopts_hamiltonian_sample_without_cloning():
    def target(x):
        return -0.5 * x.pow(2).sum()

    sampler = NUTS(
        target=target,
        x0=torch.zeros(2),
        step_size=0.1,
        warmup=10,
        max_depth=4,
        generator=_gen(0),
    )
    point_tensor = torch.tensor([0.6, 0.1])
    grad_tensor = torch.tensor([-0.6, -0.1])
    fresh = HamiltonianSample(
        point=point_tensor,
        log_prob=torch.tensor(-0.18),
        grad=grad_tensor,
    )
    point_ptr_before = point_tensor.data_ptr()
    grad_ptr_before = grad_tensor.data_ptr()

    sampler.x = fresh

    assert sampler._x is fresh
    assert sampler._x.point.data_ptr() == point_ptr_before
    assert sampler._x.grad.data_ptr() == grad_ptr_before


def test_hmc_chain_runs_with_no_clone_setter(correlated_gaussian):
    """Smoke test: after dropping the defensive clone, an HMC chain still
    converges to the right Gaussian."""

    sampler = HamiltonianMonteCarlo(
        target=correlated_gaussian,
        x0=torch.zeros(2),
        step_size=0.2,
        num_steps=5,
        generator=_gen(0),
    )
    samples, _ = sampler(1500, burn_in=300, show_progress=False)
    assert torch.allclose(samples.mean(dim=0), correlated_gaussian.mean, atol=0.25)
