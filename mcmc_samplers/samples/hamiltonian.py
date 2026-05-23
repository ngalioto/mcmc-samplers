"""HamiltonianSample carries a position, momentum, and (cached) gradient."""

from __future__ import annotations

import torch

from .sample import Sample


class HamiltonianSample(Sample):
    """Sample type used by Hamiltonian Monte Carlo and the No-U-Turn sampler.

    The position ``point`` is stored as a detached tensor; gradient computation
    is handled explicitly by the dynamics class (see
    :func:`mcmc_samplers.utils.autograd.log_prob_and_grad`).

    Attributes
    ----------
    point : torch.Tensor
        Position variable (1-D).
    log_prob : torch.Tensor or None
        Target log-density at ``point``.
    grad : torch.Tensor or None
        Gradient of ``log_prob`` w.r.t. ``point``.
    p : torch.Tensor or None
        Momentum variable (1-D), same shape as ``point``.
    """

    __slots__ = ("p",)

    def __init__(
        self,
        point: torch.Tensor | Sample | None = None,
        log_prob: torch.Tensor | None = None,
        grad: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
    ):
        if isinstance(point, Sample):
            super().__init__(
                point=point.point,
                log_prob=point.log_prob if log_prob is None else log_prob,
                grad=point.grad if grad is None else grad,
            )
        else:
            super().__init__(point=point, log_prob=log_prob, grad=grad)
        self.p = p

    def clone(self) -> "HamiltonianSample":
        base = super().clone()
        return HamiltonianSample(
            point=base.point,
            log_prob=base.log_prob,
            grad=base.grad,
            p=self.p.detach().clone() if self.p is not None else None,
        )

    def __repr__(self) -> str:
        parts = [f"point={self.point!s}"]
        if self.log_prob is not None:
            parts.append(f"log_prob={self.log_prob!s}")
        if self.p is not None:
            parts.append(f"p={self.p!s}")
        return f"HamiltonianSample({', '.join(parts)})"

    __str__ = __repr__
