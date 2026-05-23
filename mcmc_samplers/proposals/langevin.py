"""Metropolis-adjusted Langevin (MALA) proposal.

References
----------
Roberts, G.O., and Rosenthal, J.S. (1998). "Optimal Scaling of Discrete
Approximations to Langevin Diffusions."
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Sequence

import torch

from ..samples.sample import Sample
from ..utils.autograd import log_prob_and_grad
from .gaussian_random_walk import _cholesky_with_jitter
from .proposal import Proposal


class LangevinProposal(Proposal):
    """Langevin (drift + diffusion) proposal used by MALA.

    The unadjusted Langevin update is

    .. math:: y = x + 0.5 h^2 \\, A \\, \\nabla \\log\\pi(x) + h \\, L \\, \\varepsilon,

    where ``A`` is an optional positive-definite preconditioner with Cholesky
    factor ``L`` (``A = L L^T``), and ``epsilon ~ N(0, I)``. The MH acceptance
    correction recovers the correct stationary distribution. Both ``propose``
    and ``log_prob`` cache and reuse the gradient on the input ``Sample``.

    Parameters
    ----------
    target : Callable
        Target log-density function.
    step_size : float
        Langevin step size ``h``. Note the drift uses ``h^2 / 2``.
    preconditioner : torch.Tensor, optional
        Symmetric positive-definite preconditioner ``A``. Defaults to identity.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 1e-1,
        preconditioner: torch.Tensor | None = None,
    ):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.target = target
        self.step_size = float(step_size)

        if preconditioner is None:
            self._preconditioner = None
            self._sqrt_preconditioner = None
        else:
            if preconditioner.ndim != 2 or (
                preconditioner.shape[0] != preconditioner.shape[1]
            ):
                raise ValueError(
                    "preconditioner must be a square 2-D tensor; got "
                    f"{tuple(preconditioner.shape)}"
                )
            self._preconditioner = preconditioner.clone()
            self._sqrt_preconditioner = _cholesky_with_jitter(preconditioner)

    @property
    def is_symmetric(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._preconditioner is not None:
            return self._preconditioner.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        if self._preconditioner is not None:
            return self._preconditioner.dtype
        return torch.get_default_dtype()

    def _drift(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self._preconditioner is None:
            return point + 0.5 * (self.step_size ** 2) * grad
        return point + 0.5 * (self.step_size ** 2) * (self._preconditioner @ grad)

    def _ensure_grad(self, sample: Sample) -> None:
        if sample.grad is None:
            log_prob, grad = log_prob_and_grad(self.target, sample.point)
            sample.log_prob = log_prob
            sample.grad = grad

    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> Sample | list:
        if N != 1:
            raise ValueError(
                "LangevinProposal supports only one proposal per call "
                f"(got N={N})."
            )

        current = y[0]
        self._ensure_grad(current)
        mean = self._drift(current.point, current.grad)

        dim = current.point.shape[-1]
        eps = torch.randn(
            dim,
            generator=generator,
            device=current.point.device,
            dtype=current.point.dtype,
        )
        if self._sqrt_preconditioner is None:
            noise = eps
        else:
            noise = self._sqrt_preconditioner @ eps
        new_point = mean + self.step_size * noise

        # MALA always needs the gradient at ``new_point`` to evaluate the
        # reverse-direction drift in ``log_prob`` during the MH ratio. Compute
        # it together with the log-density in a single forward+backward pass
        # and cache both on the returned Sample. This avoids the wasteful
        # pattern of a forward-only target eval (for ``log_prob``) followed by
        # a separate forward+backward eval (for ``grad``).
        new_log_prob, new_grad = log_prob_and_grad(self.target, new_point)
        return Sample(point=new_point, log_prob=new_log_prob, grad=new_grad)

    def log_prob(self, y: Sequence[Sample]) -> torch.Tensor:
        """Log-density of moving from ``y[0]`` to ``y[-1]`` under the drift kernel.

        The drift uses the gradient at ``y[0]``; that gradient is cached on the
        Sample to avoid repeated autograd evaluations.
        """
        current, proposed = y[0], y[-1]
        self._ensure_grad(current)
        mean = self._drift(current.point, current.grad)
        delta = proposed.point - mean

        if self._sqrt_preconditioner is None:
            z = delta / self.step_size
        else:
            scaled = delta.unsqueeze(-1) / self.step_size
            z = torch.linalg.solve_triangular(
                self._sqrt_preconditioner, scaled, upper=False
            ).squeeze(-1)
        return -0.5 * z.pow(2).sum()

    def state_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "preconditioner": (
                None
                if self._preconditioner is None
                else self._preconditioner.detach().clone()
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_size = float(state["step_size"])
        pre = state.get("preconditioner")
        if pre is None:
            self._preconditioner = None
            self._sqrt_preconditioner = None
        else:
            self._preconditioner = pre.clone()
            self._sqrt_preconditioner = _cholesky_with_jitter(pre)
