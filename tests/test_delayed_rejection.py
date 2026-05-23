"""Tests for the delayed rejection acceptance ratio.

These tests follow the algorithm's natural call order (stage 1, then 2, then 3,
...). The implementation builds the forward-direction log-denominator and
log-acceptance incrementally across stages, so when calling ``_log_alpha``
outside of a normal ``_sample()`` step we manually seed the per-step state
``_fwd_log_denom = log pi(x_0), _fwd_log_alpha = None`` to mirror what
``_sample()`` does at the start of every MCMC step.
"""

from __future__ import annotations

import math

import torch

from mcmc_samplers import (
    DelayedRejectionAdaptiveMetropolis,
    GaussianRandomWalk,
    MetropolisHastings,
    Proposal,
    Sample,
    ScaledCovariance,
)
from mcmc_samplers.samplers.delayed_rejection import _log1m_exp, DelayedRejection


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _reset_dr_step(sampler: DelayedRejection) -> None:
    """Mimic ``Sampler._sample()`` start-of-step state for unit tests that call
    ``_log_alpha`` outside the normal sampling loop."""
    sampler._fwd_log_denom = sampler._x.log_prob.clone()
    sampler._fwd_log_alpha = None


def _mira_log_alpha(
    path: list[Sample],
    stage: int,
    proposals: list[Proposal],
) -> torch.Tensor:
    """Reference (stateless, recursive) implementation of Mira (2001) eq. 2.4.

    Exponential in ``stage``; used only as a ground-truth oracle for small
    stages (<= 3) in unit tests. Always includes all q-terms; symmetric
    proposals are handled correctly because their forward and reverse
    log-densities are numerically equal and cancel in the ratio.
    """

    def log_denom(p: list[Sample], n: int) -> torch.Tensor:
        result = p[0].log_prob.clone()
        for j in range(1, n + 1):
            result = result + proposals[j - 1].log_prob(p[: j + 1])
        for m in range(1, n):
            alpha_m = _mira_log_alpha(p[: m + 1], m, proposals)
            result = result + _log1m_exp(alpha_m)
        return result

    fwd = log_denom(path, stage)
    rev = log_denom(list(reversed(path)), stage)
    log_alpha = rev - fwd
    if torch.isnan(log_alpha):
        return torch.tensor(
            -float("inf"), device=log_alpha.device, dtype=log_alpha.dtype
        )
    return torch.minimum(
        torch.zeros((), device=log_alpha.device, dtype=log_alpha.dtype),
        log_alpha,
    )


def test_log1m_exp_matches_naive_for_safe_values():
    log_x = torch.tensor([-2.0, -0.5, -0.05, -1e-3])
    expected = torch.log1p(-torch.exp(log_x))
    assert torch.allclose(_log1m_exp(log_x), expected, atol=1e-6)


def test_one_stage_dr_reduces_to_mh(correlated_gaussian):
    """For 1-stage DR with symmetric proposal, accept ratio = MH ratio."""
    mh = MetropolisHastings(
        target=correlated_gaussian,
        x0=torch.tensor([0.0, 0.0]),
        cov=torch.eye(2),
        generator=_gen(0),
    )
    _reset_dr_step(mh)
    x = mh._x
    y1 = Sample(torch.tensor([0.5, 0.5]))
    y1.log_prob = correlated_gaussian(y1.point).detach()
    log_alpha = mh._log_alpha([x, y1], stage=1)

    expected = torch.minimum(
        torch.zeros((), dtype=log_alpha.dtype),
        y1.log_prob - x.log_prob,
    )
    assert torch.allclose(log_alpha, expected, atol=1e-6)


def test_two_stage_dr_acceptance_ratio_matches_mira_formula(correlated_gaussian):
    """For symmetric DRAM, ``alpha_2`` after stage 1 must match Mira eq. 2.4."""
    dram = DelayedRejectionAdaptiveMetropolis(
        target=correlated_gaussian,
        x0=torch.tensor([0.0, 0.0]),
        cov=torch.eye(2),
        n0=10_000,  # disable adaptation for this test
        gamma=0.01,
        generator=_gen(0),
    )
    _reset_dr_step(dram)
    x = dram._x
    y1 = Sample(torch.tensor([0.5, 0.5]))
    y1.log_prob = correlated_gaussian(y1.point).detach()
    y2 = Sample(torch.tensor([-0.3, 0.8]))
    y2.log_prob = correlated_gaussian(y2.point).detach()

    # Run stage 1 first (DR always progresses 1 -> 2 -> ...); this populates
    # the per-step forward-direction memoisation.
    alpha_1_code = dram._log_alpha([x, y1], stage=1)
    alpha_2_code = dram._log_alpha([x, y1, y2], stage=2)

    proposals = list(dram.proposals)

    # Hand check against the reference Mira oracle on the same path.
    expected_2 = _mira_log_alpha([x, y1, y2], 2, proposals)
    assert torch.allclose(alpha_2_code, expected_2, atol=1e-6)

    # And against the closed-form symmetric-DRAM expression for additional safety.
    q1_fwd = proposals[0].log_prob([x, y1])
    q1_rev = proposals[0].log_prob([y2, y1])
    alpha_1_fwd_closed = torch.minimum(
        torch.zeros((), dtype=x.log_prob.dtype),
        y1.log_prob - x.log_prob,
    )
    alpha_1_rev = torch.minimum(
        torch.zeros((), dtype=y2.log_prob.dtype),
        y1.log_prob - y2.log_prob,
    )
    assert torch.allclose(alpha_1_code, alpha_1_fwd_closed, atol=1e-6)
    one_m_fwd = _log1m_exp(alpha_1_fwd_closed)
    one_m_rev = _log1m_exp(alpha_1_rev)
    expected_closed = torch.minimum(
        torch.zeros((), dtype=x.log_prob.dtype),
        (y2.log_prob - x.log_prob)
        + (q1_rev - q1_fwd)
        + (one_m_rev - one_m_fwd),
    )
    assert torch.allclose(alpha_2_code, expected_closed, atol=1e-6)


def test_three_stage_symmetric_dr_matches_mira(correlated_gaussian):
    """Three-stage DR with symmetric Gaussian random walks at every stage."""
    p1 = GaussianRandomWalk(cov=torch.eye(2))
    p2 = GaussianRandomWalk(cov=torch.eye(2) * 0.1)
    p3 = GaussianRandomWalk(cov=torch.eye(2) * 0.01)

    class _DR3(DelayedRejection):
        pass

    sampler = _DR3(
        target=correlated_gaussian,
        x0=torch.tensor([0.0, 0.0]),
        proposals=[p1, p2, p3],
        num_stages=3,
        generator=_gen(0),
    )
    _reset_dr_step(sampler)
    x = sampler._x
    y1 = Sample(torch.tensor([0.5, 0.5]))
    y1.log_prob = correlated_gaussian(y1.point).detach()
    y2 = Sample(torch.tensor([-0.3, 0.8]))
    y2.log_prob = correlated_gaussian(y2.point).detach()
    y3 = Sample(torch.tensor([0.2, -0.4]))
    y3.log_prob = correlated_gaussian(y3.point).detach()

    # Walk through the stages in order (stage 1, 2, 3).
    sampler._log_alpha([x, y1], stage=1)
    sampler._log_alpha([x, y1, y2], stage=2)
    alpha_3 = sampler._log_alpha([x, y1, y2, y3], stage=3)

    expected = _mira_log_alpha([x, y1, y2, y3], 3, [p1, p2, p3])
    assert torch.allclose(alpha_3, expected, atol=1e-6)


class _AsymShift(Proposal):
    """Test-only asymmetric proposal: q(a -> b) = N(b; a + shift, I).

    Not invertible/standard; only ``log_prob`` and ``is_symmetric`` are needed
    here, because the tests evaluate the acceptance ratio on caller-supplied
    sample points rather than sampling from the proposal itself.
    """

    def __init__(self, shift: torch.Tensor):
        self.shift = shift.clone()

    @property
    def is_symmetric(self) -> bool:
        return False

    def propose(self, y, N=1, generator=None):  # pragma: no cover - unused
        raise NotImplementedError

    def log_prob(self, y) -> torch.Tensor:
        delta = y[-1].point - y[0].point - self.shift
        return -0.5 * delta.pow(2).sum()

    def state_dict(self) -> dict:
        return {"shift": self.shift.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        self.shift = state["shift"].clone()


def test_two_stage_asymmetric_dr_matches_mira(correlated_gaussian):
    """Both proposals asymmetric: every intermediate AND final q-ratio contributes."""
    p1 = _AsymShift(torch.tensor([0.1, -0.2]))
    p2 = _AsymShift(torch.tensor([-0.05, 0.15]))

    class _DR2(DelayedRejection):
        pass

    sampler = _DR2(
        target=correlated_gaussian,
        x0=torch.tensor([0.0, 0.0]),
        proposals=[p1, p2],
        num_stages=2,
        generator=_gen(0),
    )
    _reset_dr_step(sampler)
    x = sampler._x
    y1 = Sample(torch.tensor([0.7, -0.4]))
    y1.log_prob = correlated_gaussian(y1.point).detach()
    y2 = Sample(torch.tensor([-0.6, 0.3]))
    y2.log_prob = correlated_gaussian(y2.point).detach()

    sampler._log_alpha([x, y1], stage=1)
    alpha_2 = sampler._log_alpha([x, y1, y2], stage=2)

    expected = _mira_log_alpha([x, y1, y2], 2, [p1, p2])
    assert torch.allclose(alpha_2, expected, atol=1e-6)


def test_three_stage_mixed_symmetric_asymmetric_dr_matches_mira(correlated_gaussian):
    """Three stages, mixing symmetric and asymmetric proposals.

    The proposal at each stage is independently symmetric or asymmetric; the
    incremental implementation must handle every combination of intermediate
    vs final symmetry correctly.
    """
    p1 = GaussianRandomWalk(cov=torch.eye(2))            # symmetric
    p2 = _AsymShift(torch.tensor([0.1, 0.0]))            # asymmetric
    p3 = GaussianRandomWalk(cov=torch.eye(2) * 0.05)    # symmetric

    class _DR3(DelayedRejection):
        pass

    sampler = _DR3(
        target=correlated_gaussian,
        x0=torch.tensor([0.0, 0.0]),
        proposals=[p1, p2, p3],
        num_stages=3,
        generator=_gen(0),
    )
    _reset_dr_step(sampler)
    x = sampler._x
    y1 = Sample(torch.tensor([0.3, 0.4]))
    y1.log_prob = correlated_gaussian(y1.point).detach()
    y2 = Sample(torch.tensor([-0.2, 0.7]))
    y2.log_prob = correlated_gaussian(y2.point).detach()
    y3 = Sample(torch.tensor([0.1, -0.3]))
    y3.log_prob = correlated_gaussian(y3.point).detach()

    sampler._log_alpha([x, y1], stage=1)
    sampler._log_alpha([x, y1, y2], stage=2)
    alpha_3 = sampler._log_alpha([x, y1, y2, y3], stage=3)

    expected = _mira_log_alpha([x, y1, y2, y3], 3, [p1, p2, p3])
    assert torch.allclose(alpha_3, expected, atol=1e-6)


def test_dr_fwd_memoisation_resets_between_mcmc_steps(correlated_gaussian):
    """Verify ``_sample()`` re-initialises the per-step forward cache so that
    cross-step state never leaks (which is the safety property motivating the
    explicit reset)."""
    dram = DelayedRejectionAdaptiveMetropolis(
        target=correlated_gaussian,
        x0=torch.tensor([0.1, -0.2]),
        cov=torch.eye(2),
        n0=10_000,
        gamma=0.01,
        generator=_gen(0),
    )
    # First step.
    dram(N=1, show_progress=False)
    first_x_log_prob = dram._x.log_prob.clone()

    # Second step: the per-step cache must be re-seeded to log pi(x) at the new
    # state, not carry forward from the previous step's stage-2 (if it happened).
    dram(N=1, show_progress=False)
    assert torch.allclose(dram._fwd_log_denom, dram._x.log_prob)
    # And it cannot be the previous step's value unless the chain didn't move.
    if not torch.equal(dram._x.log_prob, first_x_log_prob):
        assert not torch.equal(dram._fwd_log_denom, first_x_log_prob)
