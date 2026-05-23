"""Sample data structures used throughout the mcmc-samplers package."""

from __future__ import annotations

import torch


class Sample:
    """Data structure representing a single MCMC sample.

    Attributes
    ----------
    point : torch.Tensor
        The sample value as a 1-D tensor.
    log_prob : torch.Tensor or None
        The (un-normalized) log-probability of the target distribution evaluated at
        ``point``. Stored as a 0-D tensor or ``None`` if not yet computed.
    grad : torch.Tensor or None
        The gradient of ``log_prob`` with respect to ``point``. Stored as a 1-D
        tensor of the same shape as ``point`` or ``None``. Populated by gradient-
        based proposals (HMC, MALA, NUTS); ignored otherwise.
    """

    __slots__ = ("point", "log_prob", "grad")

    def __init__(
        self,
        point: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
        grad: torch.Tensor | None = None,
    ):
        self.point = point
        self.log_prob = log_prob
        self.grad = grad

    @property
    def shape(self) -> torch.Size:
        return self.point.shape

    @property
    def dim(self) -> int:
        return int(self.point.shape[-1]) if self.point.ndim > 0 else 0

    @property
    def device(self) -> torch.device:
        return self.point.device

    @property
    def dtype(self) -> torch.dtype:
        return self.point.dtype

    def clone(self) -> "Sample":
        return Sample(
            point=self.point.detach().clone() if self.point is not None else None,
            log_prob=(
                self.log_prob.detach().clone() if self.log_prob is not None else None
            ),
            grad=self.grad.detach().clone() if self.grad is not None else None,
        )

    def __len__(self) -> int:
        return self.point.shape[0]

    def __repr__(self) -> str:
        parts = [f"point={self.point!s}"]
        if self.log_prob is not None:
            parts.append(f"log_prob={self.log_prob!s}")
        if self.grad is not None:
            parts.append(f"grad={self.grad!s}")
        return f"Sample({', '.join(parts)})"

    __str__ = __repr__
