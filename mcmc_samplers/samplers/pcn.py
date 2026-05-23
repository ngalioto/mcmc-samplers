"""Preconditioned Crank-Nicolson (pCN) sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import torch

from ..proposals.pcn import PreconditionedCrankNicolson
from ..samples.sample import Sample
from .delayed_rejection import DelayedRejection


class PreconditionedCrankNicolsonSampler(DelayedRejection):
    """Convenience sampler pairing the pCN proposal with MH acceptance.

    Use this when the prior is exactly the Gaussian encoded by
    ``prior_cov``/``prior_sqrt_cov`` and the ``target`` argument is the
    *likelihood* (log-density minus log-prior). The pCN proposal is
    prior-reversible, so the Metropolis ratio reduces to the likelihood ratio
    alone -- giving a dimension-independent acceptance rate.

    Parameters
    ----------
    likelihood : Callable
        Log-likelihood function (NOT the full posterior log-density).
    x0 : Sample or torch.Tensor
        Initial chain state.
    prior_cov, prior_sqrt_cov : torch.Tensor
        Prior covariance, supplied as either ``prior_cov`` or its lower Cholesky
        factor. Exactly one must be provided.
    beta : float
        Step size in ``(0, 1]``. Smaller values give higher acceptance.
    """

    def __init__(
        self,
        likelihood: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        prior_cov: torch.Tensor | None = None,
        prior_sqrt_cov: torch.Tensor | None = None,
        beta: float = 0.5,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        super().__init__(
            target=likelihood,
            x0=x0,
            proposals=[
                PreconditionedCrankNicolson(
                    prior_cov=prior_cov, prior_sqrt_cov=prior_sqrt_cov, beta=beta
                )
            ],
            num_stages=1,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )
