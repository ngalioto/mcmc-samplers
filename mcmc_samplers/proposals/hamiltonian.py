"""Hamiltonian dynamics proposal for HMC and NUTS.

The proposal integrates Hamilton's equations with the leapfrog scheme, given
a target log-density (the negative potential energy) and an auxiliary momentum
sampled from a Gaussian whose covariance is the user-specified mass matrix.
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Sequence

import torch

from ..samples.hamiltonian import HamiltonianSample
from ..samples.sample import Sample
from ..utils.autograd import log_prob_and_grad
from .gaussian_random_walk import _cholesky_with_jitter
from .proposal import Proposal


class HamiltonianDynamics(Proposal):
    """Proposal that integrates Hamiltonian dynamics with leapfrog.

    Parameters
    ----------
    target : Callable
        Log-density of the target distribution. Equivalently, the negative
        potential energy ``-U(q)``.
    dim : int
        Dimensionality of the position variable.
    step_size : float
        Leapfrog step size (commonly ``epsilon``).
    num_steps : int
        Number of leapfrog steps per proposal (commonly ``L``).
    mass : torch.Tensor or None
        Mass matrix ``M`` used for the kinetic energy
        ``K(p) = 0.5 * p^T M^{-1} p``. ``None`` (default) means identity.
        A 1-D tensor is interpreted as a diagonal mass; a 2-D tensor as a
        full matrix. The momentum sample distribution is ``N(0, M)``.
    device, dtype : optional
        Device and dtype for sampled momenta. Inferred from ``mass`` when
        provided.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        step_size: float = 1e-1,
        num_steps: int = 1,
        mass: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.target = target
        self.dim = int(dim)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)

        if mass is None:
            self._device = device if device is not None else torch.device("cpu")
            self._dtype = dtype if dtype is not None else torch.get_default_dtype()
            self._mass = None
            self._mass_inv = None
            self._sqrt_mass = None
        else:
            self._device = device if device is not None else mass.device
            self._dtype = dtype if dtype is not None else mass.dtype
            self._set_mass(mass)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def is_symmetric(self) -> bool:
        return False

    @property
    def mass(self) -> torch.Tensor | None:
        return None if self._mass is None else self._mass.clone()

    def set_mass(self, mass: torch.Tensor | None) -> None:
        """Update the mass matrix; ``None`` resets to identity."""
        if mass is None:
            self._mass = None
            self._mass_inv = None
            self._sqrt_mass = None
            return
        self._set_mass(mass)

    def _set_mass(self, mass: torch.Tensor) -> None:
        mass = mass.to(device=self._device, dtype=self._dtype)
        if mass.ndim == 1:
            if mass.shape[0] != self.dim:
                raise ValueError(
                    f"diagonal mass must have shape ({self.dim},), got "
                    f"{tuple(mass.shape)}"
                )
            if torch.any(mass <= 0):
                raise ValueError("diagonal mass entries must be positive")
            self._mass = mass.clone()
            self._mass_inv = 1.0 / mass
            self._sqrt_mass = mass.sqrt()
        elif mass.ndim == 2:
            if mass.shape != (self.dim, self.dim):
                raise ValueError(
                    f"mass matrix must be ({self.dim}, {self.dim}), got "
                    f"{tuple(mass.shape)}"
                )
            self._mass = mass.clone()
            chol = _cholesky_with_jitter(mass)
            self._sqrt_mass = chol
            identity = torch.eye(self.dim, device=self._device, dtype=self._dtype)
            self._mass_inv = torch.cholesky_solve(identity, chol, upper=False)
        else:
            raise ValueError("mass must be 1-D (diagonal) or 2-D (full).")

    def _sample_momentum(
        self, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        eps = torch.randn(
            self.dim, generator=generator, device=self._device, dtype=self._dtype
        )
        if self._mass is None:
            return eps
        if self._sqrt_mass.ndim == 1:
            return self._sqrt_mass * eps
        return self._sqrt_mass @ eps

    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """Return ``0.5 * p^T M^{-1} p`` (the kinetic energy)."""
        if self._mass_inv is None:
            return 0.5 * p.pow(2).sum()
        if self._mass_inv.ndim == 1:
            return 0.5 * (self._mass_inv * p * p).sum()
        return 0.5 * (p @ self._mass_inv @ p)

    def _momentum_to_velocity(self, p: torch.Tensor) -> torch.Tensor:
        if self._mass_inv is None:
            return p
        if self._mass_inv.ndim == 1:
            return self._mass_inv * p
        return self._mass_inv @ p

    def leapfrog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        step_size: float | None = None,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate Hamilton's equations forward ``num_steps`` leapfrog steps.

        Parameters
        ----------
        q : torch.Tensor
            Initial position (1-D, will be detached internally).
        p : torch.Tensor
            Initial momentum (1-D).
        grad : torch.Tensor
            Gradient of ``target`` at ``q``.
        step_size : float, optional
            Override the configured ``step_size``.
        num_steps : int, optional
            Override the configured ``num_steps``.

        Returns
        -------
        q, p, log_prob, grad : torch.Tensor
            New position, new momentum, log-density at new position, and the
            gradient at the new position.
        """
        h = self.step_size if step_size is None else float(step_size)
        L = self.num_steps if num_steps is None else int(num_steps)

        q = q.detach().clone()
        p = p.detach().clone()
        grad = grad.detach().clone()

        p = p + 0.5 * h * grad
        log_prob = None
        for ii in range(L):
            q = q + h * self._momentum_to_velocity(p)
            log_prob, grad = log_prob_and_grad(self.target, q)
            if ii != L - 1:
                p = p + h * grad
        p = p + 0.5 * h * grad

        return q, p, log_prob, grad

    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> HamiltonianSample:
        """Sample a momentum and integrate from the current position.

        Only ``N == 1`` is supported by Hamiltonian dynamics (one trajectory
        per proposal). ``y[0]`` must be a ``HamiltonianSample`` with ``log_prob``
        and ``grad`` populated.
        """
        if N != 1:
            raise ValueError(
                "HamiltonianDynamics only supports a single proposal per call "
                f"(got N={N})."
            )

        current = y[0]
        if current.grad is None:
            log_prob, grad = log_prob_and_grad(self.target, current.point)
            current.log_prob = log_prob
            current.grad = grad

        p0 = self._sample_momentum(generator=generator)
        current.p = p0

        q_new, p_new, log_prob_new, grad_new = self.leapfrog(
            current.point, p0, current.grad
        )

        return HamiltonianSample(
            point=q_new,
            log_prob=log_prob_new,
            grad=grad_new,
            p=p_new,
        )

    def log_prob(self, y: Sequence[Sample]) -> torch.Tensor:
        """Log-density of the proposal kernel q(y[0] -> y[-1]).

        For HMC the proposal density is determined by the momentum sampled at
        the starting point: q(y[0] -> y[-1]) is proportional to N(p_0 | 0, M),
        where p_0 is the momentum carried by ``y[0]``. The choice of momentum
        sign cancels because the kinetic energy is even. Returning
        ``-K(y[0].p)`` (instead of ``-K(y[-1].p)``) is what makes the resulting
        Metropolis ratio reduce to the standard Hamiltonian-difference form
        ``log_pi(q_1) - log_pi(q_0) + K(p_0) - K(p_1)``.
        """
        return -self.kinetic_energy(y[0].p)

    def state_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "num_steps": self.num_steps,
            "mass": None if self._mass is None else self._mass.detach().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_size = float(state["step_size"])
        self.num_steps = int(state["num_steps"])
        self.set_mass(state.get("mass"))
