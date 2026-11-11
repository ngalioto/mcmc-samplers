"""Preconditioned Crank-Nicolson (pCN) proposal.

References
----------
Cotter, S.L., Roberts, G.O., Stuart, A.M., and White, D. (2013).
"MCMC methods for functions: modifying old algorithms to make them faster."
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch

from ..samples.sample import Sample
from .gaussian_random_walk import _cholesky_with_jitter
from .proposal import Proposal


class PreconditionedCrankNicolson(Proposal):
    """Dimension-robust, prior-reversible proposal for Gaussian-prior targets.

    The proposal is

    .. math:: y = \\sqrt{1 - \\beta^2} \\, x + \\beta \\, L \\, \\varepsilon

    where ``L`` is a Cholesky factor of the prior covariance and ``epsilon ~
    N(0, I)``. The proposal is reversible with respect to the prior ``N(0, C)``,
    so when paired with the standard MH acceptance kernel using only the
    likelihood (not the prior) as the un-normalized log-density, the algorithm
    targets the posterior.

    The proposal is not symmetric, so in a generic MH sampler the target must
    be the full posterior. For the likelihood-only acceptance rule use
    :class:`~mcmc_samplers.PreconditionedCrankNicolsonSampler`.

    Parameters
    ----------
    prior_sqrt_cov : torch.Tensor, optional
        Lower Cholesky factor of the prior covariance ``C``. Either this or
        ``prior_cov`` must be provided.
    prior_cov : torch.Tensor, optional
        Prior covariance ``C`` (symmetric positive-definite).
    beta : float
        Step size in ``(0, 1]``. Smaller values give higher acceptance.
    """

    def __init__(
        self,
        prior_sqrt_cov: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
        beta: float = 0.5,
    ):
        if prior_sqrt_cov is None and prior_cov is None:
            raise ValueError(
                "PreconditionedCrankNicolson requires prior_sqrt_cov or prior_cov."
            )
        if prior_sqrt_cov is not None and prior_cov is not None:
            raise ValueError(
                "Pass exactly one of prior_sqrt_cov / prior_cov, not both."
            )
        if not (0 < beta <= 1):
            raise ValueError(f"beta must lie in (0, 1], got {beta}")

        if prior_sqrt_cov is not None:
            if prior_sqrt_cov.ndim != 2 or (
                prior_sqrt_cov.shape[0] != prior_sqrt_cov.shape[1]
            ):
                raise ValueError(
                    "prior_sqrt_cov must be a square 2-D tensor; got "
                    f"{tuple(prior_sqrt_cov.shape)}"
                )
            self.prior_sqrt_cov = prior_sqrt_cov.clone()
        else:
            self.prior_sqrt_cov = _cholesky_with_jitter(prior_cov)

        self.dim = int(self.prior_sqrt_cov.shape[0])
        self.beta = float(beta)

    @property
    def is_symmetric(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        return self.prior_sqrt_cov.device

    @property
    def dtype(self) -> torch.dtype:
        return self.prior_sqrt_cov.dtype

    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> Sample | list:
        center = y[0].point
        rho = sqrt(1.0 - self.beta ** 2)
        eps = torch.randn(
            N, self.dim, generator=generator, device=self.device, dtype=self.dtype
        )
        noise = eps @ self.prior_sqrt_cov.T
        sample_points = rho * center + self.beta * noise
        if N == 1:
            return Sample(sample_points.squeeze(0))
        return [Sample(sample_points[ii]) for ii in range(N)]

    def log_prob(self, y: Sequence[Sample]) -> torch.Tensor:
        rho = sqrt(1.0 - self.beta ** 2)
        delta = (y[-1].point - rho * y[0].point).unsqueeze(-1) / self.beta
        z = torch.linalg.solve_triangular(self.prior_sqrt_cov, delta, upper=False)
        return -0.5 * z.pow(2).sum()

    def state_dict(self) -> dict:
        return {
            "prior_sqrt_cov": self.prior_sqrt_cov.detach().clone(),
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict) -> None:
        self.prior_sqrt_cov = state["prior_sqrt_cov"].clone()
        self.beta = float(state["beta"])
        self.dim = int(self.prior_sqrt_cov.shape[0])
