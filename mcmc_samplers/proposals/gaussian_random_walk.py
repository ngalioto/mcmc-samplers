"""Gaussian random-walk proposal family.

Includes the basic random walk, the adaptive-covariance variant (Haario et al.
2001) using Welford-style online statistics, and a scaled wrapper used by the
delayed-rejection algorithm.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence, Union

import torch

from ..samples.sample import Sample
from .proposal import Proposal


def _cholesky_with_jitter(
    matrix: torch.Tensor,
    eps: float = 1e-8,
    max_tries: int = 9,
) -> torch.Tensor:
    """Cholesky factorize ``matrix``, adding increasing diagonal jitter on failure.

    Tries ``matrix``, ``matrix + 1e-8 I``, ``matrix + 1e-7 I``, ..., up to
    ``max_tries`` orders of magnitude before giving up. Always uses the
    lower-triangular factor.

    Returns
    -------
    torch.Tensor
        Lower-triangular Cholesky factor ``L`` such that ``L @ L.T == matrix``.

    Raises
    ------
    torch.linalg.LinAlgError
        If even the maximally-jittered matrix is not positive-definite.
    """
    try:
        return torch.linalg.cholesky(matrix, upper=False)
    except RuntimeError:
        pass

    dim = matrix.shape[-1]
    identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    for ii in range(max_tries):
        jitter = eps * (10 ** ii)
        try:
            return torch.linalg.cholesky(matrix + jitter * identity, upper=False)
        except RuntimeError:
            continue
    raise torch.linalg.LinAlgError(
        "Cholesky decomposition failed even after jittering up to "
        f"{eps * 10 ** (max_tries - 1):.2e} on the diagonal."
    )


class GaussianRandomWalk(Proposal):
    """Gaussian random-walk proposal centered on the current chain state.

    Parameters
    ----------
    sqrt_cov : torch.Tensor, optional
        Lower Cholesky factor of the proposal covariance. Must be square.
        Exactly one of ``sqrt_cov`` or ``cov`` must be provided.
    cov : torch.Tensor, optional
        Symmetric positive-definite proposal covariance.
    """

    def __init__(
        self,
        sqrt_cov: torch.Tensor | None = None,
        cov: torch.Tensor | None = None,
    ):
        if sqrt_cov is not None and cov is not None:
            raise ValueError("Provide exactly one of sqrt_cov or cov, not both.")
        if sqrt_cov is None and cov is None:
            raise ValueError("GaussianRandomWalk requires either sqrt_cov or cov.")

        if sqrt_cov is not None:
            if sqrt_cov.ndim != 2 or sqrt_cov.shape[0] != sqrt_cov.shape[1]:
                raise ValueError(
                    "sqrt_cov must be a square 2D matrix, got shape "
                    f"{tuple(sqrt_cov.shape)}."
                )
            self.sqrt_cov = sqrt_cov.clone()
        else:
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError(
                    f"cov must be a square 2D matrix, got shape {tuple(cov.shape)}."
                )
            self.sqrt_cov = _cholesky_with_jitter(cov)

        self.dim = int(self.sqrt_cov.shape[0])

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self.sqrt_cov.device

    @property
    def dtype(self) -> torch.dtype:
        return self.sqrt_cov.dtype

    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> Union[Sample, list]:
        """Draw ``N`` samples from a Gaussian centered at ``y[0].point``.

        ``y[0].point`` may be ``(dim,)`` or ``(num_chains, dim)``. Returns a
        single :class:`Sample` if ``N == 1`` and a list otherwise.
        """
        center = y[0].point
        eps = torch.randn(
            N, *center.shape, generator=generator, device=self.device, dtype=self.dtype
        )
        sample_points = eps @ self.sqrt_cov.T + center
        if N == 1:
            return Sample(sample_points.squeeze(0))
        return [Sample(sample_points[ii]) for ii in range(N)]

    def log_prob(self, y: Sequence[Sample]) -> torch.Tensor:
        """Un-normalized log-density of moving from ``y[0]`` to ``y[-1]``
        (one value per chain when batched)."""
        delta = (y[-1].point - y[0].point).unsqueeze(-1)
        z = torch.linalg.solve_triangular(self.sqrt_cov, delta, upper=False)
        return -0.5 * z.pow(2).sum(dim=(-2, -1))

    def state_dict(self) -> dict:
        return {"sqrt_cov": self.sqrt_cov.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        self.sqrt_cov = state["sqrt_cov"].clone()
        self.dim = int(self.sqrt_cov.shape[0])


class AdaptiveCovariance(GaussianRandomWalk):
    """Gaussian random walk with covariance adapted from the chain history.

    Implements the recursive sample covariance of Haario et al. (2001) using a
    numerically-stable, exactly-symmetric Welford-style update of the
    second-order moment. The proposal covariance is held fixed at
    ``sd * cov_0`` until ``_iteration`` exceeds ``n0``; afterward it becomes
    ``sd * (M2 / (n - 1) + eps * I)``.

    Symmetric Welford update
        On step ``n``, with the pre-update mean ``mean_{n-1}`` and new sample
        ``x``, the standard two-pass Welford increment ``outer(delta, delta2)``
        (where ``delta2 = x - mean_n``) can be rewritten using
        ``delta2 = (n - 1)/n * delta`` as ``((n - 1) / n) * outer(delta, delta)``.
        This single rank-one update is symmetric by construction (it cannot
        drift off-symmetric due to floating-point round-off) and saves a
        per-step outer product.

    Identity cache
        After ``n0``, every ``_adapt`` call adds ``eps * I`` to the empirical
        covariance and Cholesky-decomposes it. Constructing the identity each
        time is cheap but redundant; ``cache_identity=True`` (the default)
        keeps a single ``dim x dim`` tensor alongside the running statistics.
        Set ``cache_identity=False`` to trade the extra ``dim^2`` floats of
        storage for a fresh allocation each step. (The empirical covariance
        itself is intentionally *not* cached because scalar multiplication is
        cheap and a second persistent matrix would double the storage cost.)

    Adapt interval
        ``adapt_interval=1`` (the default) recomputes ``sqrt_cov`` every
        post-warmup step so that adaptation stays as responsive as possible
        and the proposal covariance does not drift between recomputations.
        Larger values amortize the ``O(dim^3)`` Cholesky over multiple
        steps -- recommended only when the Cholesky is a measurable
        bottleneck (e.g. very high-dimensional problems).

    Parameters
    ----------
    cov : torch.Tensor
        Initial proposal covariance used while ``_iteration <= n0``.
    sd : float, optional
        Global scale factor. Defaults to ``2.4 ** 2 / dim`` (Roberts et al. 2001).
    n0 : int
        Number of samples to observe before adaptation kicks in.
    eps : float
        Small positive value added to the empirical covariance diagonal to
        keep it positive-definite.
    cache_identity : bool, optional
        If True (default), keep a single cached ``dim x dim`` identity matrix
        to avoid re-allocating it on every ``_adapt`` call. Set to False to
        save ``dim^2`` floats of memory at a tiny per-iteration cost.
    adapt_interval : int, optional
        Recompute the Cholesky factor every ``adapt_interval`` post-warmup
        steps. Default ``1`` (recompute every step). Must be ``>= 1``.
    """

    def __init__(
        self,
        cov: torch.Tensor,
        sd: float | None = None,
        n0: int = 200,
        eps: float = 1e-8,
        cache_identity: bool = True,
        adapt_interval: int = 1,
    ):
        super().__init__(cov=cov)
        if adapt_interval < 1:
            raise ValueError(
                f"adapt_interval must be >= 1, got {adapt_interval}"
            )
        self.sd = (2.4 ** 2) / self.dim if sd is None else float(sd)
        self.sqrt_cov = self.sqrt_cov * sqrt(self.sd)
        self._cov_0 = cov.clone() * self.sd
        self.n0 = int(n0)
        self.eps = float(eps)
        self.cache_identity = bool(cache_identity)
        self.adapt_interval = int(adapt_interval)

        self._iteration = 0
        self._mean = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        self._M2 = torch.zeros(
            self.dim, self.dim, device=self.device, dtype=self.dtype
        )
        self._identity = (
            torch.eye(self.dim, device=self.device, dtype=self.dtype)
            if self.cache_identity
            else None
        )

    @property
    def iteration(self) -> int:
        """Number of adaptation steps performed so far (read-only)."""
        return self._iteration

    @property
    def mean(self) -> torch.Tensor:
        """Running sample mean (read-only clone)."""
        return self._mean.clone()

    def _eye(self) -> torch.Tensor:
        """Return a ``dim x dim`` identity, cached if ``cache_identity=True``."""
        if self._identity is not None:
            return self._identity
        return torch.eye(self.dim, device=self.device, dtype=self.dtype)

    @property
    def cov(self) -> torch.Tensor:
        """Current proposal covariance.

        Returns the (scaled) initial covariance until ``_iteration > n0``;
        afterwards returns the (scaled, jittered) empirical sample covariance.
        """
        if self._iteration > self.n0 and self._iteration > 1:
            empirical = self._M2 / (self._iteration - 1)
            return self.sd * (empirical + self.eps * self._eye())
        return self._cov_0

    def _adapt(self, x: torch.Tensor) -> None:
        """Update running mean and second-moment with the new sample ``x``.

        ``x`` may be one sample ``(dim,)`` or one per chain ``(num_chains, dim)``;
        all chains then share (and speed up) the adaptation. Refreshes
        ``sqrt_cov`` each time another ``adapt_interval`` samples past ``n0``
        have been seen.
        """
        x = x.detach().reshape(-1, self.dim)
        m = x.shape[0]
        n_old = self._iteration
        n = n_old + m
        self._iteration = n
        # Merge the batch's statistics into the running ones (Chan et al. 1979).
        # For one sample this is the symmetric Welford update
        # M2 += ((n-1)/n) * outer(delta, delta): exactly symmetric, one outer.
        batch_mean = x.mean(dim=0)
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * m / n
        if m > 1:
            centered = x - batch_mean
            self._M2 = self._M2 + centered.T @ centered
        if n_old > 0:
            self._M2 = self._M2 + (n_old * m / n) * torch.outer(delta, delta)

        k = self.adapt_interval
        if n > self.n0 and (n - self.n0) // k > (max(n_old, self.n0) - self.n0) // k:
            try:
                self.sqrt_cov = _cholesky_with_jitter(self.cov, eps=self.eps)
            except torch.linalg.LinAlgError:
                pass

    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> Union[Sample, list]:
        return super().propose(y, N=N, generator=generator)

    def state_dict(self) -> dict:
        return {
            "sqrt_cov": self.sqrt_cov.detach().clone(),
            "cov_0": self._cov_0.detach().clone(),
            "iteration": self._iteration,
            "mean": self._mean.detach().clone(),
            "M2": self._M2.detach().clone(),
            "sd": self.sd,
            "n0": self.n0,
            "eps": self.eps,
            "cache_identity": self.cache_identity,
            "adapt_interval": self.adapt_interval,
        }

    def load_state_dict(self, state: dict) -> None:
        self.sqrt_cov = state["sqrt_cov"].clone()
        self._cov_0 = state["cov_0"].clone()
        self._iteration = int(state["iteration"])
        self._mean = state["mean"].clone()
        self._M2 = state["M2"].clone()
        self.sd = float(state["sd"])
        self.n0 = int(state["n0"])
        self.eps = float(state["eps"])
        self.cache_identity = bool(state.get("cache_identity", True))
        self.adapt_interval = int(state.get("adapt_interval", 1))
        self.dim = int(self.sqrt_cov.shape[0])
        self._identity = (
            torch.eye(self.dim, device=self.device, dtype=self.dtype)
            if self.cache_identity
            else None
        )


class ScaledCovariance(GaussianRandomWalk):
    """Gaussian random walk whose covariance is a scaled view of another's.

    Used by delayed rejection: stage-2 covariance is ``gamma`` times the
    stage-1 (typically adaptive) covariance. Because this only reads from the
    underlying proposal, scale adaptation is automatic.

    Parameters
    ----------
    covariance : GaussianRandomWalk
        Underlying proposal whose ``sqrt_cov`` is read on demand.
    gamma : float
        Scale factor applied to the underlying ``sqrt_cov`` (covariance is
        scaled by ``gamma`` so the Cholesky factor is scaled by ``sqrt(gamma)``).
    """

    def __init__(self, covariance: GaussianRandomWalk, gamma: float = 1e-2):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.covariance = covariance
        self.gamma = float(gamma)
        self.sqrt_gamma = sqrt(self.gamma)

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def sqrt_cov(self) -> torch.Tensor:
        return self.covariance.sqrt_cov * self.sqrt_gamma

    @property
    def dim(self) -> int:
        return self.covariance.dim

    @property
    def device(self) -> torch.device:
        return self.covariance.device

    @property
    def dtype(self) -> torch.dtype:
        return self.covariance.dtype

    def state_dict(self) -> dict:
        return {"gamma": self.gamma}

    def load_state_dict(self, state: dict) -> None:
        self.gamma = float(state["gamma"])
        self.sqrt_gamma = sqrt(self.gamma)
