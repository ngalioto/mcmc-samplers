"""Metropolis-adjusted Langevin algorithm (MALA) sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import torch

from ..proposals.langevin import LangevinProposal
from ..samples.sample import Sample
from ..utils.autograd import log_prob_and_grad
from .delayed_rejection import DelayedRejection


class MetropolisAdjustedLangevin(DelayedRejection):
    """Metropolis-adjusted Langevin algorithm (Roberts and Tweedie 1996).

    Uses :class:`~mcmc_samplers.proposals.langevin.LangevinProposal` for the
    gradient-driven proposal and the standard Metropolis-Hastings acceptance
    kernel.

    Parameters
    ----------
    target : Callable
        Log-density of the target distribution.
    x0 : Sample or torch.Tensor
        Initial chain state.
    step_size : float
        Langevin step size ``h``.
    preconditioner : torch.Tensor, optional
        Symmetric positive-definite preconditioner for the drift. Defaults to
        identity.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        step_size: float = 1e-1,
        preconditioner: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        super().__init__(
            target=target,
            x0=x0,
            proposals=[
                LangevinProposal(
                    target=target,
                    step_size=step_size,
                    preconditioner=preconditioner,
                )
            ],
            num_stages=1,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )

    @property
    def x(self) -> Sample:
        return self._x

    @x.setter
    def x(self, value: Union[Sample, torch.Tensor]) -> None:
        """Set the chain state, eagerly populating both ``log_prob`` and ``grad``.

        Overrides the base setter to consolidate the initial forward-only
        ``_eval_target`` call and the subsequent forward+backward pass into a
        single ``log_prob_and_grad`` evaluation. This ensures iteration 1's
        propose call does no wasted target evaluations.
        """
        if isinstance(value, Sample):
            point = value.point.detach()
            log_prob = value.log_prob
            grad = value.grad
        elif isinstance(value, torch.Tensor):
            point = value.detach()
            log_prob = None
            grad = None
        else:
            raise TypeError(
                f"x must be a Sample or torch.Tensor, got {type(value).__name__}"
            )
        if point.ndim != 1:
            point = point.reshape(-1)

        if log_prob is None or grad is None:
            log_prob, grad = log_prob_and_grad(self.target, point)

        if isinstance(value, Sample):
            value.point = point
            value.log_prob = log_prob
            value.grad = grad
            self._x = value
        else:
            self._x = Sample(point=point, log_prob=log_prob, grad=grad)

    def _reeval_log_prob(self) -> None:
        """Refresh both ``log_prob`` and ``grad`` at the current point."""
        log_prob, grad = log_prob_and_grad(self.target, self._x.point)
        self._x.log_prob = log_prob
        self._x.grad = grad
