"""No-U-Turn Sampler (NUTS) with multinomial sampling and dual-averaging.

References
----------
Hoffman, M.D. and Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo." JMLR 15: 1593-1623.

Betancourt, M. (2016). "Identifying the Optimal Integration Time in
Hamiltonian Monte Carlo." arXiv:1601.00225.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from math import log
from typing import Union

import torch

from ..proposals.hamiltonian import HamiltonianDynamics
from ..samples.hamiltonian import HamiltonianSample
from ..samples.sample import Sample
from ..utils.autograd import log_prob_and_grad
from .sampler import Sampler


class _SubTree:
    """Container for build-tree return values (multinomial NUTS)."""

    __slots__ = (
        "q_minus",
        "p_minus",
        "grad_minus",
        "q_plus",
        "p_plus",
        "grad_plus",
        "q_sample",
        "log_p_sample",
        "grad_sample",
        "log_weight",
        "diverged",
        "u_turn",
        "alpha_sum",
        "n_alpha",
    )

    def __init__(
        self,
        q_minus,
        p_minus,
        grad_minus,
        q_plus,
        p_plus,
        grad_plus,
        q_sample,
        log_p_sample,
        grad_sample,
        log_weight,
        diverged,
        u_turn,
        alpha_sum,
        n_alpha,
    ):
        self.q_minus = q_minus
        self.p_minus = p_minus
        self.grad_minus = grad_minus
        self.q_plus = q_plus
        self.p_plus = p_plus
        self.grad_plus = grad_plus
        self.q_sample = q_sample
        self.log_p_sample = log_p_sample
        self.grad_sample = grad_sample
        self.log_weight = log_weight
        self.diverged = diverged
        self.u_turn = u_turn
        self.alpha_sum = alpha_sum
        self.n_alpha = n_alpha


def _logsumexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(exp(a) + exp(b))``."""
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


class NUTS(Sampler):
    """No-U-Turn Sampler (Hoffman & Gelman 2014; Betancourt 2016).

    Implements the multinomial-sampling variant of NUTS with dual-averaging
    step-size adaptation during a warmup phase. Each call to ``_sample``
    runs one NUTS iteration: it doubles a balanced binary tree of leapfrog
    states up to ``max_depth`` or until the no-U-turn (or divergence) criterion
    fires, then samples one state from the trajectory weighted by
    ``exp(-H)``.

    Parameters
    ----------
    target : Callable
        Log-density of the target distribution.
    x0 : Sample or torch.Tensor
        Initial chain state.
    step_size : float
        Initial leapfrog step size. Adapted during warmup.
    max_depth : int
        Maximum tree depth (``2 ** max_depth`` is the largest possible
        trajectory length).
    target_accept : float
        Desired average Metropolis-Hastings acceptance, used by dual averaging.
    max_energy_change : float
        Trajectories where the energy changes by more than this are flagged
        as diverging and the tree is terminated early.
    warmup : int
        Number of initial iterations during which the step size is adapted.
        After ``warmup`` steps, ``step_size`` is frozen at the adapted value.
    mass : torch.Tensor, optional
        Mass matrix for the kinetic energy. See
        :class:`~mcmc_samplers.proposals.hamiltonian.HamiltonianDynamics`.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        step_size: float = 0.25,
        max_depth: int = 10,
        target_accept: float = 0.8,
        max_energy_change: float = 1000.0,
        warmup: int = 1000,
        mass: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        point = x0.point if isinstance(x0, Sample) else x0
        if point.ndim != 1:
            point = point.reshape(-1)
        dim = int(point.shape[-1])

        dynamics = HamiltonianDynamics(
            target=target,
            dim=dim,
            step_size=step_size,
            num_steps=1,
            mass=mass,
            device=point.device,
            dtype=point.dtype,
        )
        super().__init__(
            target=target,
            x0=x0,
            proposals=[dynamics],
            acceptance_kernels=[
                lambda y: torch.zeros((), dtype=point.dtype, device=point.device)
            ],
            generator=generator,
            noisy=False,
            reeval=0,
        )

        self.max_depth = int(max_depth)
        self.target_accept = float(target_accept)
        self.max_energy_change = float(max_energy_change)
        self.warmup = int(warmup)

        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75
        self._mu = log(10.0 * step_size)
        self._H_bar = 0.0
        self._log_eps_bar = 0.0
        self._adapt_step = 0
        self.last_tree_depth = 0
        self.last_alpha = 1.0
        self.last_diverged = False

    @property
    def dynamics(self) -> HamiltonianDynamics:
        return self.proposals[0]

    @property
    def step_size(self) -> float:
        return self.dynamics.step_size

    @property
    def x(self) -> HamiltonianSample:
        return self._x

    @x.setter
    def x(self, value: Union[torch.Tensor, Sample]) -> None:
        # See ``HamiltonianMonteCarlo.x`` setter: the hot path adopts the
        # already-detached HamiltonianSample produced internally; callers that
        # want to retain ownership should clone before assigning.
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
                f"x must be a Sample or torch.Tensor, got {type(value).__name__}"
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

    def _sample_momentum(self) -> torch.Tensor:
        return self.dynamics._sample_momentum(generator=self.generator)

    def _hamiltonian(self, q_log_prob: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return -q_log_prob + self.dynamics.kinetic_energy(p)

    def _leaf(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        H0: torch.Tensor,
        direction: int,
    ) -> _SubTree:
        """Single leapfrog step in direction ``+1`` or ``-1``; one tree leaf."""
        step = direction * self.step_size
        q_new, p_new, log_prob_new, grad_new = self.dynamics.leapfrog(
            q, p, grad, step_size=step, num_steps=1
        )
        H_new = self._hamiltonian(log_prob_new, p_new)
        log_weight = H0 - H_new
        diverged = bool((H_new - H0).abs().item() > self.max_energy_change)

        accept_prob = torch.exp(torch.clamp(log_weight, max=0.0))
        return _SubTree(
            q_minus=q_new,
            p_minus=p_new,
            grad_minus=grad_new,
            q_plus=q_new,
            p_plus=p_new,
            grad_plus=grad_new,
            q_sample=q_new,
            log_p_sample=log_prob_new,
            grad_sample=grad_new,
            log_weight=log_weight.detach(),
            diverged=diverged,
            u_turn=False,
            alpha_sum=accept_prob.detach(),
            n_alpha=1,
        )

    def _u_turn(
        self,
        q_minus: torch.Tensor,
        p_minus: torch.Tensor,
        q_plus: torch.Tensor,
        p_plus: torch.Tensor,
    ) -> bool:
        delta = q_plus - q_minus
        v_minus = self.dynamics._momentum_to_velocity(p_minus)
        v_plus = self.dynamics._momentum_to_velocity(p_plus)
        return bool((delta @ v_minus).item() < 0.0 or (delta @ v_plus).item() < 0.0)

    def _build_tree(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        H0: torch.Tensor,
        direction: int,
        depth: int,
    ) -> _SubTree:
        if depth == 0:
            return self._leaf(q, p, grad, H0, direction)

        left = self._build_tree(q, p, grad, H0, direction, depth - 1)
        if left.diverged or left.u_turn:
            return left

        if direction == -1:
            right = self._build_tree(
                left.q_minus, left.p_minus, left.grad_minus, H0, direction, depth - 1
            )
        else:
            right = self._build_tree(
                left.q_plus, left.p_plus, left.grad_plus, H0, direction, depth - 1
            )

        combined_log_weight = _logsumexp(left.log_weight, right.log_weight)
        if torch.isfinite(combined_log_weight):
            log_prob_pick_right = right.log_weight - combined_log_weight
            u = torch.rand(
                (),
                generator=self.generator,
                device=combined_log_weight.device,
                dtype=combined_log_weight.dtype,
            )
            pick_right = torch.log(u) < log_prob_pick_right
        else:
            pick_right = False

        if pick_right:
            q_sample = right.q_sample
            log_p_sample = right.log_p_sample
            grad_sample = right.grad_sample
        else:
            q_sample = left.q_sample
            log_p_sample = left.log_p_sample
            grad_sample = left.grad_sample

        if direction == -1:
            q_minus, p_minus, grad_minus = right.q_minus, right.p_minus, right.grad_minus
            q_plus, p_plus, grad_plus = left.q_plus, left.p_plus, left.grad_plus
        else:
            q_minus, p_minus, grad_minus = left.q_minus, left.p_minus, left.grad_minus
            q_plus, p_plus, grad_plus = right.q_plus, right.p_plus, right.grad_plus

        u_turn = (
            right.u_turn
            or self._u_turn(q_minus, p_minus, q_plus, p_plus)
        )

        return _SubTree(
            q_minus=q_minus,
            p_minus=p_minus,
            grad_minus=grad_minus,
            q_plus=q_plus,
            p_plus=p_plus,
            grad_plus=grad_plus,
            q_sample=q_sample,
            log_p_sample=log_p_sample,
            grad_sample=grad_sample,
            log_weight=combined_log_weight,
            diverged=right.diverged,
            u_turn=u_turn,
            alpha_sum=left.alpha_sum + right.alpha_sum,
            n_alpha=left.n_alpha + right.n_alpha,
        )

    def _adapt_step_size(self, alpha: float) -> None:
        self._adapt_step += 1
        m = float(self._adapt_step)
        weight = 1.0 / (m + self._t0)
        self._H_bar = (1.0 - weight) * self._H_bar + weight * (
            self.target_accept - alpha
        )
        log_eps = self._mu - math.sqrt(m) / self._gamma * self._H_bar
        eta = m ** (-self._kappa)
        self._log_eps_bar = eta * log_eps + (1.0 - eta) * self._log_eps_bar
        self.dynamics.step_size = math.exp(log_eps)

    def _sample(self) -> int:
        q0 = self._x.point
        p0 = self._sample_momentum()
        grad0 = self._x.grad
        log_p0 = self._x.log_prob

        H0 = self._hamiltonian(log_p0, p0)

        q_minus = q_plus = q0
        p_minus = p_plus = p0
        grad_minus = grad_plus = grad0
        q_sample = q0
        log_p_sample = log_p0
        grad_sample = grad0
        log_total_weight = torch.zeros(
            (), device=q0.device, dtype=q0.dtype
        )

        depth = 0
        diverged = False
        u_turn = False
        alpha_sum = torch.zeros((), device=q0.device, dtype=q0.dtype)
        n_alpha = 0

        while depth < self.max_depth and not u_turn and not diverged:
            direction = 1 if torch.rand(
                (),
                generator=self.generator,
                device=q0.device,
                dtype=q0.dtype,
            ).item() < 0.5 else -1
            if direction == -1:
                subtree = self._build_tree(
                    q_minus, p_minus, grad_minus, H0, direction, depth
                )
            else:
                subtree = self._build_tree(
                    q_plus, p_plus, grad_plus, H0, direction, depth
                )

            alpha_sum = alpha_sum + subtree.alpha_sum
            n_alpha += subtree.n_alpha

            if not subtree.diverged and not subtree.u_turn:
                log_combined = _logsumexp(log_total_weight, subtree.log_weight)
                if torch.isfinite(log_combined):
                    log_prob_pick = subtree.log_weight - log_combined
                    u = torch.rand(
                        (),
                        generator=self.generator,
                        device=q0.device,
                        dtype=q0.dtype,
                    )
                    if torch.log(u) < log_prob_pick:
                        q_sample = subtree.q_sample
                        log_p_sample = subtree.log_p_sample
                        grad_sample = subtree.grad_sample
                log_total_weight = log_combined

            if direction == -1:
                q_minus, p_minus, grad_minus = (
                    subtree.q_minus,
                    subtree.p_minus,
                    subtree.grad_minus,
                )
            else:
                q_plus, p_plus, grad_plus = (
                    subtree.q_plus,
                    subtree.p_plus,
                    subtree.grad_plus,
                )

            if subtree.diverged:
                diverged = True
                break
            if subtree.u_turn or self._u_turn(q_minus, p_minus, q_plus, p_plus):
                u_turn = True
            depth += 1

        self.last_tree_depth = depth
        self.last_diverged = diverged
        alpha = (
            float(alpha_sum.item() / n_alpha) if n_alpha > 0 else 0.0
        )
        self.last_alpha = alpha

        moved = not torch.equal(q_sample, q0)
        # ``q_sample``, ``log_p_sample`` and ``grad_sample`` are already
        # detached outputs of ``leapfrog`` (or the prior ``self._x`` if the
        # trajectory never moved off the origin), and nothing else holds a
        # reference to them, so we adopt them directly instead of paying for
        # three detach+clone operations per step.
        self._x = HamiltonianSample(
            point=q_sample,
            log_prob=log_p_sample,
            grad=grad_sample,
        )

        if self._adapt_step < self.warmup:
            self._adapt_step_size(alpha)
            if self._adapt_step >= self.warmup:
                self.dynamics.step_size = math.exp(self._log_eps_bar)

        return 1 if moved else 0

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "step_size": self.dynamics.step_size,
                "H_bar": self._H_bar,
                "log_eps_bar": self._log_eps_bar,
                "adapt_step": self._adapt_step,
                "warmup": self.warmup,
                "max_depth": self.max_depth,
                "target_accept": self.target_accept,
                "max_energy_change": self.max_energy_change,
                "mu": self._mu,
            }
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.dynamics.step_size = float(state["step_size"])
        self._H_bar = float(state["H_bar"])
        self._log_eps_bar = float(state["log_eps_bar"])
        self._adapt_step = int(state["adapt_step"])
        self.warmup = int(state["warmup"])
        self.max_depth = int(state["max_depth"])
        self.target_accept = float(state["target_accept"])
        self.max_energy_change = float(state["max_energy_change"])
        self._mu = float(state["mu"])
