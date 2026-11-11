"""Parallel-tempering / replica-exchange sampler."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Union

import torch

from ..samples.sample import Sample
from .sampler import Sampler, _default_generator


class ParallelTempering(Sampler):
    """Parallel tempering (replica exchange) sampler.

    Runs ``len(betas)`` independent chains in parallel, with each replica
    targeting the *tempered* density ``pi(x) ** beta``. After every MCMC step,
    a swap between a uniformly chosen adjacent pair of replicas is proposed
    and accepted with probability

    .. math:: \\min(1, \\exp((\\beta_j - \\beta_i)\\,(\\log\\pi(x_i) - \\log\\pi(x_j)))).

    The cold chain (``beta = 1``) is the chain that targets the original
    density; only its samples are returned by ``__call__``. Swap acceptance
    statistics are exposed via :attr:`swap_acceptance_ratios`.

    Parameters
    ----------
    target : Callable
        Original (un-tempered) log-density.
    x0 : Sample or torch.Tensor
        Initial chain state used for every replica.
    sampler_factory : Callable[[Callable, x0], Sampler]
        Factory that builds one replica. Signature
        ``factory(tempered_target, x0) -> Sampler``. Use a lambda to bind
        sampler-specific parameters (e.g. step size).
    betas : sequence of float
        Inverse temperatures, in any order. Must contain ``1.0`` exactly once
        and all be strictly positive. The recommended schedule is a geometric
        ladder, e.g. ``(1.0, 0.5, 0.25, 0.125)``.
    swap_every : int
        Propose a swap only on every ``swap_every``-th iteration.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        sampler_factory: Callable[
            [Callable[[torch.Tensor], torch.Tensor], Union[Sample, torch.Tensor]],
            Sampler,
        ],
        betas: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
        swap_every: int = 1,
        generator: torch.Generator | None = None,
    ):
        betas = tuple(float(b) for b in betas)
        if any(b <= 0 for b in betas):
            raise ValueError("All betas must be strictly positive.")
        if 1.0 not in betas:
            raise ValueError(
                "betas must contain 1.0 (the cold chain). Got " f"{betas}."
            )

        self.target = target
        self.betas = betas
        self.swap_every = int(swap_every)
        self._cold_idx = betas.index(1.0)
        if generator is None:
            x0_point = x0.point if isinstance(x0, Sample) else x0
            generator = _default_generator(x0_point.device)
        self.generator = generator

        self.replicas: list[Sampler] = []
        for beta in self.betas:
            tempered = self._make_tempered_target(target, beta)
            replica = sampler_factory(tempered, x0)
            self.replicas.append(replica)

        self.proposals: tuple = ()
        self.acceptance_kernels: tuple = ()
        self.noisy = False
        self.reeval = 0
        self.batched = False
        self.acceptance_ratio = 0.0
        self._step = 0
        self.swap_acceptance_ratios = [0.0] * (len(self.betas) - 1)
        self._swap_attempts = [0] * (len(self.betas) - 1)

    @staticmethod
    def _make_tempered_target(
        target: Callable[[torch.Tensor], torch.Tensor], beta: float
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def tempered(x: torch.Tensor) -> torch.Tensor:
            return beta * target(x)

        return tempered

    @property
    def cold(self) -> Sampler:
        """The cold-chain replica (``beta = 1``)."""
        return self.replicas[self._cold_idx]

    @property
    def x(self) -> Sample:
        return self.cold.x

    @x.setter
    def x(self, value) -> None:
        raise NotImplementedError(
            "ParallelTempering does not allow direct assignment to x; "
            "set each replica's state via the replicas list."
        )

    @property
    def _x(self) -> Sample:
        # Sampler.__call__ reads self._x; the chain state is the cold replica's.
        return self.cold._x

    @property
    def dim(self) -> int:
        return self.cold.dim

    @property
    def device(self) -> torch.device:
        return self.cold.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cold.dtype

    def _reeval_log_prob(self) -> None:
        for replica in self.replicas:
            replica._reeval_log_prob()

    def _swap_state(self, i: int, j: int) -> None:
        """Swap the chain states between replicas ``i`` and ``j``.

        Re-tempers the cached ``log_prob`` and ``grad`` (when present) by a
        scalar factor rather than re-evaluating ``target`` -- this is exact
        because both quantities are linear in ``beta``.
        """
        beta_i, beta_j = self.betas[i], self.betas[j]

        xi = self.replicas[i].x.clone()
        xj = self.replicas[j].x.clone()

        if xi.log_prob is not None:
            xi.log_prob = (beta_j / beta_i) * xi.log_prob
        if xi.grad is not None:
            xi.grad = (beta_j / beta_i) * xi.grad

        if xj.log_prob is not None:
            xj.log_prob = (beta_i / beta_j) * xj.log_prob
        if xj.grad is not None:
            xj.grad = (beta_i / beta_j) * xj.grad

        self.replicas[i]._x = xj
        self.replicas[j]._x = xi

    def _propose_swap(self) -> tuple[int, int, bool]:
        """Propose a swap between a uniformly chosen adjacent replica pair.

        Returns ``(pair_index, cold_changed, accepted)``.
        """
        n = len(self.replicas)
        if n < 2:
            return -1, False, False

        i = int(
            torch.randint(
                0,
                n - 1,
                (1,),
                generator=self.generator,
                device=torch.device("cpu"),
            ).item()
        )
        j = i + 1

        beta_i, beta_j = self.betas[i], self.betas[j]
        log_pi_i = self.replicas[i].x.log_prob.detach() / beta_i
        log_pi_j = self.replicas[j].x.log_prob.detach() / beta_j

        log_alpha = (beta_j - beta_i) * (log_pi_i - log_pi_j)
        if torch.isnan(log_alpha):
            log_alpha = torch.tensor(
                -float("inf"), device=log_alpha.device, dtype=log_alpha.dtype
            )
        log_alpha = torch.minimum(torch.zeros_like(log_alpha), log_alpha)

        u = torch.rand(
            (),
            generator=self.generator,
            device=log_alpha.device,
            dtype=log_alpha.dtype,
        )
        accepted = bool(torch.log(u).item() < float(log_alpha.item()))

        cold_changed = accepted and (i == self._cold_idx or j == self._cold_idx)
        if accepted:
            self._swap_state(i, j)

        self._swap_attempts[i] += 1
        attempts = self._swap_attempts[i]
        ar = self.swap_acceptance_ratios
        ar[i] = ((attempts - 1) * ar[i] + (1 if accepted else 0)) / attempts

        return i, cold_changed, accepted

    def _sample(self) -> int:
        cold_accepted_own = 0
        for k, replica in enumerate(self.replicas):
            acc = replica._sample()
            if k == self._cold_idx:
                cold_accepted_own = acc

        cold_accepted_swap = 0
        if self._step % self.swap_every == 0 and len(self.replicas) > 1:
            _, cold_changed, _ = self._propose_swap()
            if cold_changed:
                cold_accepted_swap = 1

        return 1 if (cold_accepted_own or cold_accepted_swap) else 0

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "acceptance_ratio": self.acceptance_ratio,
            "swap_acceptance_ratios": list(self.swap_acceptance_ratios),
            "swap_attempts": list(self._swap_attempts),
            "generator": self.generator.get_state(),
            "replicas": [r.state_dict() for r in self.replicas],
            "betas": list(self.betas),
            "swap_every": self.swap_every,
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state["betas"]) != tuple(self.betas):
            raise ValueError("betas mismatch between state_dict and current sampler")
        self._step = int(state["step"])
        self.acceptance_ratio = float(state["acceptance_ratio"])
        self.swap_acceptance_ratios = list(state["swap_acceptance_ratios"])
        self._swap_attempts = list(state["swap_attempts"])
        self.generator.set_state(state["generator"])
        self.swap_every = int(state["swap_every"])
        for replica, r_state in zip(self.replicas, state["replicas"]):
            replica.load_state_dict(r_state)
