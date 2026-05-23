"""Delayed rejection samplers and the classic Metropolis-Hastings variants.

References
----------
Green, P.J. and Mira, A. (2001). "Delayed rejection in reversible jump
Metropolis-Hastings." Biometrika 88(4): 1035-1053.

Haario, H., et al. (2006). "DRAM: efficient adaptive MCMC." Statistics and
Computing 16: 339-354.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Sequence, Union

import torch

from ..proposals.gaussian_random_walk import (
    AdaptiveCovariance,
    GaussianRandomWalk,
    ScaledCovariance,
)
from ..proposals.proposal import Proposal
from ..samples.sample import Sample
from .sampler import Sampler


def _log1m_exp(log_x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(log_x))`` for ``log_x <= 0``.

    Uses ``log1p(-exp(.))`` for small ``|log_x|`` and ``log(-expm1(.))`` for
    larger ``|log_x|``, in line with the Wolfram / R-stats recommendation.
    Returns ``-inf`` when ``log_x == 0``.
    """
    log_half = torch.log(torch.tensor(0.5, dtype=log_x.dtype, device=log_x.device))
    return torch.where(
        log_x > log_half,
        torch.log(-torch.expm1(log_x)),
        torch.log1p(-torch.exp(log_x)),
    )


class DelayedRejection(Sampler, ABC):
    """Multi-stage delayed rejection sampler (Green & Mira, 2001).

    Implements Mira (2001) eq. 2.4. The acceptance ratio at stage ``n`` is

    .. math::
       \\log\\alpha_n = \\log N_n - \\log D_n,

    with

    .. math::
       \\log D_n = \\log\\pi(x_0)
                 + \\sum_{j=1}^n \\log q_j^{\\text{fwd}}
                 + \\sum_{m=1}^{n-1} \\log(1 - \\alpha_m^{\\text{fwd}}),

    and :math:`\\log N_n` is the same expression evaluated on the reversed
    path :math:`(x_n, x_{n-1}, \\ldots, x_0)`. Symmetric kernels at the
    *last* stage are skipped because :math:`q_n^{\\text{fwd}}` and
    :math:`q_n^{\\text{rev}}` cancel exactly; symmetric kernels at
    *intermediate* stages still need explicit q-evals because their
    conditioning paths differ between the forward and reverse directions.

    Efficient incremental evaluation
        The forward log-denominator :math:`\\log D_n` only depends on the
        forward sub-path, so its value is reused from stage to stage within
        a single MCMC step:
        :math:`\\log D_n = \\log D_{n-1} + \\log(1 - \\alpha_{n-1}) +
        \\log q_n^{\\text{fwd}}`.
        These cached values live in ``_fwd_log_denom`` and
        ``_fwd_log_alpha`` and are reset by :meth:`_sample` at the start
        of every step (they carry no information across MCMC steps and are
        deliberately omitted from :meth:`state_dict`).

        :math:`\\log N_n` is structurally identical to :math:`\\log D_n` on
        the reversed path, so it is built iteratively for
        ``k = 1, \\ldots, n`` rather than evaluated by independent
        recursive calls; each sub-stage :math:`\\alpha_k` is computed
        once using exactly the same incremental machinery. This keeps the
        per-step work :math:`O(n)` proposal log-density evaluations -- the
        same as Green & Mira's original derivation -- while leaving the
        recursion stateless from the caller's perspective.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        proposals: Sequence[Proposal],
        num_stages: int | None = None,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        if num_stages is None:
            num_stages = len(proposals)
        if num_stages != len(proposals):
            raise ValueError(
                "num_stages must equal len(proposals) for delayed rejection "
                f"(got {num_stages} and {len(proposals)})."
            )

        acceptance_kernels = [
            (lambda y, stage=stage_idx: self._log_alpha(y, stage))
            for stage_idx in range(1, num_stages + 1)
        ]
        super().__init__(
            target=target,
            x0=x0,
            proposals=proposals,
            acceptance_kernels=acceptance_kernels,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )

        # Per-step memoization of the forward (outer) log-denominator and
        # log-acceptance. Reset at the start of each MCMC step; intentionally
        # not serialised in state_dict (they encode no information between
        # steps).
        self._fwd_log_denom: torch.Tensor | None = None
        self._fwd_log_alpha: torch.Tensor | None = None

    def _sample(self) -> int:
        # Reset forward-direction memoization. ``_fwd_log_denom`` is
        # initialised to ``log pi(x_0)`` (the empty-path denominator);
        # ``_fwd_log_alpha`` is ``None`` until stage 1 finishes.
        self._fwd_log_denom = self._x.log_prob.clone()
        self._fwd_log_alpha = None
        return super()._sample()

    def _log_alpha(self, y: Sequence[Sample], stage: int) -> torch.Tensor:
        """Compute log-acceptance for stage ``stage`` on the forward path ``y``.

        Updates ``self._fwd_log_denom`` and ``self._fwd_log_alpha`` so the
        next stage's call can pick up the incremental forward state.
        """
        assert len(y) == stage + 1, (
            f"expected len(y) == {stage + 1}, got {len(y)}"
        )

        log_alpha, log_denom = self._stage_log_alpha(
            y, stage, self._fwd_log_denom, self._fwd_log_alpha
        )
        self._fwd_log_denom = log_denom
        self._fwd_log_alpha = log_alpha
        return log_alpha

    def _stage_log_alpha(
        self,
        y: Sequence[Sample],
        stage: int,
        prev_log_denom: torch.Tensor,
        prev_log_alpha: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log alpha_stage, log D_stage)`` on path ``y``.

        ``prev_log_denom`` and ``prev_log_alpha`` are the log-denominator and
        log-acceptance from stage ``stage - 1`` *on this same path*. For the
        top-level call these are read from the per-step forward memoisation;
        for inner reverse-direction recursions they are local accumulators
        passed explicitly (no shared mutable state across the recursion).
        """
        if y[-1].log_prob is None:
            y[-1].log_prob = self._eval_target(y[-1].point)

        # Forward log-denominator: D_stage = D_{stage-1} + log(1 - alpha_{stage-1})
        # + intermediate symmetric q-term (added here when the previous-stage
        # proposal is symmetric because that q was suppressed at the previous
        # stage; the proposal's q now appears as an *intermediate* term and no
        # longer cancels).
        log_denom = prev_log_denom.clone()
        if stage > 1:
            log_denom = log_denom + _log1m_exp(prev_log_alpha)
            if self.proposals[stage - 2].is_symmetric:
                log_denom = log_denom + self.proposals[stage - 2].log_prob(
                    y[: stage]
                )

        # Reverse log-numerator: iteratively build D_k on the reversed path
        # (y[stage], y[stage-1], ..., y[0]) for k = 1, 2, ..., stage, threading
        # the running (log_denom_rev, log_alpha_rev) through the loop. Each
        # sub-stage's alpha is one recursive call into ``_stage_log_alpha`` --
        # so the recursion is linear in ``stage``, not exponential.
        rev_log_denom = y[stage].log_prob.clone()
        rev_log_alpha: torch.Tensor | None = None
        for k in range(1, stage):
            sub_rev = y[stage : stage - k - 1 : -1]
            k_log_alpha, k_log_denom = self._stage_log_alpha(
                sub_rev, k, rev_log_denom, rev_log_alpha
            )
            rev_log_denom = k_log_denom
            rev_log_alpha = k_log_alpha

        log_numer = rev_log_denom
        if stage > 1:
            log_numer = log_numer + _log1m_exp(rev_log_alpha)
            if self.proposals[stage - 2].is_symmetric:
                log_numer = log_numer + self.proposals[stage - 2].log_prob(
                    y[stage : 0 : -1]
                )

        # Last-stage proposal q-terms. Symmetric kernels give
        # q_stage^{fwd} = q_stage^{rev}, which cancel and are skipped.
        if not self.proposals[stage - 1].is_symmetric:
            log_denom = log_denom + self.proposals[stage - 1].log_prob(y)
            log_numer = log_numer + self.proposals[stage - 1].log_prob(y[::-1])

        log_alpha = log_numer - log_denom
        if torch.isnan(log_alpha):
            log_alpha = torch.tensor(
                -float("inf"), device=log_alpha.device, dtype=log_alpha.dtype
            )
        log_alpha = torch.minimum(
            torch.zeros((), device=log_alpha.device, dtype=log_alpha.dtype),
            log_alpha,
        )
        return log_alpha, log_denom


class MetropolisHastings(DelayedRejection):
    """Random-walk Metropolis-Hastings sampler.

    Parameters
    ----------
    target : Callable
        Log-density of the target distribution.
    x0 : Sample or torch.Tensor
        Initial chain state.
    sqrt_cov : torch.Tensor, optional
        Lower Cholesky factor of the Gaussian random-walk covariance.
    cov : torch.Tensor, optional
        Random-walk covariance. Exactly one of ``sqrt_cov`` / ``cov`` must be
        passed.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        sqrt_cov: torch.Tensor | None = None,
        cov: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        super().__init__(
            target=target,
            x0=x0,
            proposals=[GaussianRandomWalk(sqrt_cov=sqrt_cov, cov=cov)],
            num_stages=1,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )


class AdaptiveMetropolis(DelayedRejection):
    """Adaptive Metropolis sampler (Haario et al. 2001).

    Updates the proposal covariance online using the running sample covariance
    of the chain (symmetric Welford recursion). Adaptation begins after ``n0``
    iterations.

    Parameters
    ----------
    cache_identity : bool, optional
        Forwarded to :class:`AdaptiveCovariance`. Default ``True``.
    adapt_interval : int, optional
        Forwarded to :class:`AdaptiveCovariance`. Default ``1``.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        cov: torch.Tensor,
        sd: float | None = None,
        n0: int = 200,
        eps: float = 1e-8,
        cache_identity: bool = True,
        adapt_interval: int = 1,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        super().__init__(
            target=target,
            x0=x0,
            proposals=[
                AdaptiveCovariance(
                    cov=cov,
                    sd=sd,
                    n0=n0,
                    eps=eps,
                    cache_identity=cache_identity,
                    adapt_interval=adapt_interval,
                )
            ],
            num_stages=1,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )

    def _sample(self) -> int:
        self.proposals[0]._adapt(self._x.point)
        return super()._sample()


class DelayedRejectionAdaptiveMetropolis(DelayedRejection):
    """Delayed rejection adaptive Metropolis (DRAM) sampler (Haario et al. 2006).

    Combines two-stage delayed rejection with adaptive covariance. The
    stage-2 covariance is a scaled view of the (adapting) stage-1 covariance.

    Parameters
    ----------
    cache_identity : bool, optional
        Forwarded to :class:`AdaptiveCovariance`. Default ``True``.
    adapt_interval : int, optional
        Forwarded to :class:`AdaptiveCovariance`. Default ``1``.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        cov: torch.Tensor,
        sd: float | None = None,
        gamma: float = 1e-2,
        n0: int = 200,
        eps: float = 1e-8,
        cache_identity: bool = True,
        adapt_interval: int = 1,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        adaptive = AdaptiveCovariance(
            cov=cov,
            sd=sd,
            n0=n0,
            eps=eps,
            cache_identity=cache_identity,
            adapt_interval=adapt_interval,
        )
        scaled = ScaledCovariance(adaptive, gamma=gamma)
        super().__init__(
            target=target,
            x0=x0,
            proposals=[adaptive, scaled],
            num_stages=2,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )

    def _sample(self) -> int:
        self.proposals[0]._adapt(self._x.point)
        return super()._sample()
