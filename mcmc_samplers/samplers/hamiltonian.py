"""Hamiltonian Monte Carlo sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import torch

from ..proposals.hamiltonian import HamiltonianDynamics
from ..samples.hamiltonian import HamiltonianSample
from ..samples.sample import Sample
from ..utils.autograd import log_prob_and_grad
from .delayed_rejection import DelayedRejection


class HamiltonianMonteCarlo(DelayedRejection):
    """Hamiltonian Monte Carlo (Duane et al. 1987; Neal 2011).

    Pairs a Hamiltonian-dynamics proposal with the Metropolis-Hastings
    acceptance kernel. The chain state is a :class:`HamiltonianSample`
    carrying both position and a cached gradient of the log-density.

    Parameters
    ----------
    target : Callable
        Log-density of the target distribution.
    x0 : Sample or torch.Tensor
        Initial position. A bare tensor is interpreted as the position; the
        target and its gradient are computed automatically.
    step_size : float
        Leapfrog step size.
    num_steps : int
        Number of leapfrog steps per proposal.
    mass : torch.Tensor, optional
        Mass matrix for the kinetic energy. ``None`` (default) means identity.
        1-D tensor is a diagonal mass; 2-D is a full mass matrix.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        step_size: float = 1e-1,
        num_steps: int = 1,
        mass: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
    ):
        point = x0.point if isinstance(x0, Sample) else x0
        if point.ndim != 1:
            point = point.reshape(-1)
        dim = int(point.shape[-1])

        super().__init__(
            target=target,
            x0=x0,
            proposals=[
                HamiltonianDynamics(
                    target=target,
                    dim=dim,
                    step_size=step_size,
                    num_steps=num_steps,
                    mass=mass,
                    device=point.device,
                    dtype=point.dtype,
                )
            ],
            num_stages=1,
            generator=generator,
            noisy=noisy,
            reeval=reeval,
        )

    @property
    def x(self) -> HamiltonianSample:
        return self._x

    @x.setter
    def x(self, value: Union[torch.Tensor, Sample]) -> None:
        # The hot path is ``self.x = y[-1]`` after an accepted proposal: ``y[-1]``
        # is a freshly-built HamiltonianSample with detached point/log_prob/grad
        # and is not held by anyone else. Adopting it directly avoids a clone of
        # three tensors per accepted step. Callers that want to retain ownership
        # of a HamiltonianSample they pass in should ``.clone()`` it themselves.
        if isinstance(value, HamiltonianSample):
            new = value
        elif isinstance(value, Sample):
            new = HamiltonianSample(
                point=value.point,
                log_prob=value.log_prob,
                grad=value.grad,
            )
        elif isinstance(value, torch.Tensor):
            new = HamiltonianSample(point=value.detach())
        else:
            raise TypeError(
                f"x must be a (Hamiltonian)Sample or torch.Tensor, got "
                f"{type(value).__name__}"
            )

        point = new.point
        if point.ndim != 1:
            point = point.reshape(-1)
            new.point = point

        if new.log_prob is None or new.grad is None:
            log_prob, grad = log_prob_and_grad(self.target, point)
            new.log_prob = log_prob
            new.grad = grad
        self._x = new

    def _reeval_log_prob(self) -> None:
        """Re-evaluate the target log-density AND gradient at the current point.

        Overrides the base class to keep ``self._x.grad`` consistent with
        ``self._x.log_prob``. Without this, the next leapfrog would use a
        stale gradient (silent bias).
        """
        log_prob, grad = log_prob_and_grad(self.target, self._x.point)
        self._x.log_prob = log_prob
        self._x.grad = grad
