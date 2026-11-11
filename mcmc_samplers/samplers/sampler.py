"""Sampler abstract base class for MCMC algorithms."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Sequence, Tuple, Union

import torch
from tqdm import tqdm

from ..proposals.proposal import Proposal
from ..samples.sample import Sample


AcceptanceKernel = Callable[[Sequence[Sample]], torch.Tensor]


def _default_generator(device: torch.device) -> torch.Generator:
    """Fresh generator seeded from torch's global RNG.

    A bare ``torch.Generator()`` always starts from the same fixed seed, so
    unseeded chains would be identical and ignore ``torch.manual_seed``.
    """
    seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    return torch.Generator(device=device).manual_seed(seed)


class Sampler(ABC):
    """Abstract base class for MCMC samplers.

    A sampler is a (proposal, acceptance-kernel) pipeline plus chain state.
    Concrete subclasses implement specific algorithms by composing one or more
    proposals and acceptance kernels.

    Parameters
    ----------
    target : Callable
        Function mapping a 1-D tensor of shape ``(dim,)`` to a 0-D log-density.
    x0 : Sample or torch.Tensor
        Initial chain state. A bare tensor is interpreted as the point and the
        target is called to populate ``log_prob``.
    proposals : sequence of Proposal
        Ordered list of proposals (multiple stages = delayed rejection).
    acceptance_kernels : sequence of callables
        One per proposal stage. Each receives the current ``[x, y_1, ...]`` and
        returns a 0-D log-acceptance ratio (i.e. ``log(min(1, ratio))``).
    generator : torch.Generator, optional
        Per-sampler RNG. When ``None``, one is seeded from torch's global RNG
        (so ``torch.manual_seed`` makes runs reproducible).
    noisy : bool
        For stochastic likelihoods (e.g. synthetic or particle-filter
        likelihoods). If true, re-evaluate ``target`` at the current point
        after ``reeval`` consecutive rejections, so the chain does not stay
        stuck on a lucky overestimate (Maraia et al. 2023, following noisy MH,
        Medina-Aguayo et al. 2016). This improves mixing but makes the sampler
        approximate. With ``noisy=False`` the estimate at the current point is
        never refreshed, which is exact pseudo-marginal MCMC when ``exp(target)``
        is an unbiased likelihood estimate.
    reeval : int
        Number of consecutive rejections before triggering a re-evaluation.
    batched : bool
        Run several independent chains at once. ``x0`` then has shape
        ``(num_chains, dim)`` and ``target`` must map ``(num_chains, dim)`` to
        ``(num_chains,)`` (``torch.func.vmap(target)`` turns a per-point target
        into one). Each chain accepts or rejects on its own, and ``__call__``
        returns samples of shape ``(num_chains, kept, dim)``.
    """

    def __init__(
        self,
        target: Callable[[torch.Tensor], torch.Tensor],
        x0: Union[Sample, torch.Tensor],
        proposals: Sequence[Proposal],
        acceptance_kernels: Sequence[AcceptanceKernel],
        generator: torch.Generator | None = None,
        noisy: bool = False,
        reeval: int = 200,
        batched: bool = False,
    ):
        if len(proposals) != len(acceptance_kernels):
            raise ValueError(
                "proposals and acceptance_kernels must have the same length"
            )

        self.target = target
        self.proposals = tuple(proposals)
        self.acceptance_kernels = tuple(acceptance_kernels)
        if generator is None:
            x0_point = x0.point if isinstance(x0, Sample) else x0
            generator = _default_generator(x0_point.device)
        self.generator = generator
        self.noisy = bool(noisy)
        self.reeval = int(reeval)
        self.batched = bool(batched)

        self.acceptance_ratio = 0.0
        self._step = 0
        self.x = x0

    @property
    def x(self) -> Sample:
        return self._x

    def _as_point(self, point: torch.Tensor) -> torch.Tensor:
        """Shape a state as ``(dim,)``, or ``(num_chains, dim)`` when batched."""
        if self.batched:
            if point.ndim != 2:
                raise ValueError(
                    f"batched samplers need x of shape (num_chains, dim), got {tuple(point.shape)}"
                )
            return point
        return point if point.ndim == 1 else point.reshape(-1)

    @x.setter
    def x(self, value: Union[Sample, torch.Tensor]) -> None:
        if isinstance(value, Sample):
            point = self._as_point(value.point)
            self._x = value
            self._x.point = point
            if self._x.log_prob is None:
                self._x.log_prob = self._eval_target(point)
        elif isinstance(value, torch.Tensor):
            point = self._as_point(value)
            log_prob = self._eval_target(point)
            self._x = Sample(point=point.detach(), log_prob=log_prob)
        else:
            raise TypeError(
                f"x must be a Sample or torch.Tensor, got {type(value).__name__}"
            )

    def _eval_target(self, point: torch.Tensor) -> torch.Tensor:
        """Evaluate the target at ``point`` (no autograd graph).

        Returns a 0-D tensor, or one value per chain when batched. Gradient-based
        samplers get gradients via ``log_prob_and_grad``.
        """
        with torch.no_grad():
            lp = self.target(point).detach()
        if self.batched:
            return lp.reshape(point.shape[0])
        return lp.squeeze() if lp.ndim > 0 else lp

    @property
    def dim(self) -> int:
        return int(self._x.point.shape[-1])

    @property
    def device(self) -> torch.device:
        return self._x.point.device

    @property
    def dtype(self) -> torch.dtype:
        return self._x.point.dtype

    def _log_rand(self) -> torch.Tensor:
        shape = (self._x.point.shape[0],) if self.batched else (1,)
        u = torch.rand(shape, generator=self.generator, device=self.device, dtype=self.dtype)
        return torch.log(u)

    def _reeval_log_prob(self) -> None:
        """Re-evaluate the target log-density at the current point.

        Subclasses override to also refresh cached gradients (e.g. HMC).
        """
        self._x.log_prob = self._eval_target(self._x.point)

    def _sample(self) -> int | torch.Tensor:
        """Advance the chain by one MCMC step. Returns 1 on accept, 0 on reject
        (a boolean per chain when batched)."""
        if self.batched:
            return self._sample_batched()
        y: list[Sample] = [self._x]
        for proposal, acceptance_kernel in zip(
            self.proposals, self.acceptance_kernels
        ):
            y.append(proposal.propose(y, generator=self.generator))
            alpha = acceptance_kernel(y)
            if alpha > self._log_rand():
                self.x = y[-1]
                return 1
        return 0

    def _sample_batched(self) -> torch.Tensor:
        """One step for every chain. A chain moves to the proposal of the first
        stage it accepts; later stages are ignored for that chain."""
        y: list[Sample] = [self._x]
        point, log_prob = self._x.point, self._x.log_prob
        accepted = torch.zeros(point.shape[0], dtype=torch.bool, device=self.device)
        for proposal, acceptance_kernel in zip(
            self.proposals, self.acceptance_kernels
        ):
            y.append(proposal.propose(y, generator=self.generator))
            alpha = acceptance_kernel(y)
            take = ~accepted & (alpha > self._log_rand())
            point = torch.where(take[:, None], y[-1].point, point)
            log_prob = torch.where(take, y[-1].log_prob, log_prob)
            accepted = accepted | take
            if accepted.all():
                break
        self._x = Sample(point=point, log_prob=log_prob)
        return accepted

    def __call__(
        self,
        N: int = 1,
        burn_in: int = 0,
        thin: int = 1,
        callback: Callable[[int, Sample, int], None] | None = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the chain.

        Parameters
        ----------
        N : int
            Total number of MCMC steps to take, including burn-in.
        burn_in : int
            Number of leading samples to discard. Must satisfy ``0 <= burn_in < N``.
        thin : int
            Keep every ``thin``-th post-burn-in sample. Must be ``>= 1``.
        callback : callable, optional
            ``callback(step, sample, accepted)`` invoked after every MCMC step
            (including burn-in). ``step`` is the 0-indexed iteration, ``sample``
            is the chain state after the step, and ``accepted`` is 0/1.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        samples : torch.Tensor
            Tensor of kept samples, shape ``((N - burn_in) // thin, dim)``, or
            ``(num_chains, (N - burn_in) // thin, dim)`` when batched.
        log_probs : torch.Tensor
            Target log-density at each kept sample, shape
            ``((N - burn_in) // thin,)``, or ``(num_chains, ...)`` when batched.
        """
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if burn_in < 0 or burn_in >= N:
            raise ValueError(f"burn_in must satisfy 0 <= burn_in < N, got {burn_in}")
        if thin < 1:
            raise ValueError(f"thin must be >= 1, got {thin}")

        rejection_streak = (
            torch.zeros(self._x.point.shape[0], dtype=torch.long, device=self.device)
            if self.batched
            else 0
        )
        kept = max((N - burn_in + thin - 1) // thin, 0)
        samples = torch.zeros(kept, *self._x.point.shape, device=self.device, dtype=self.dtype)
        log_probs = torch.zeros(kept, *self._x.log_prob.shape, device=self.device, dtype=self.dtype)

        iterator = range(N)
        if show_progress:
            iterator = tqdm(iterator, total=N)

        keep_idx = 0
        for ii in iterator:
            accepted = self._sample()
            self._step += 1
            rate = accepted.float().mean().item() if self.batched else accepted
            self.acceptance_ratio = (
                (self._step - 1) * self.acceptance_ratio + rate
            ) / self._step

            if self.noisy and self.batched:
                # Per-chain streaks; refresh only the chains that are stuck.
                rejection_streak = torch.where(accepted, 0, rejection_streak + 1)
                stuck = rejection_streak >= self.reeval
                if stuck.any():
                    fresh = self._eval_target(self._x.point)
                    self._x.log_prob = torch.where(stuck, fresh, self._x.log_prob)
                    rejection_streak = torch.where(stuck, 0, rejection_streak)
            elif self.noisy:
                rejection_streak = 0 if accepted else rejection_streak + 1
                if rejection_streak == self.reeval:
                    self._reeval_log_prob()
                    rejection_streak = 0

            if callback is not None:
                callback(ii, self._x, accepted)

            if ii >= burn_in and ((ii - burn_in) % thin == 0) and keep_idx < kept:
                samples[keep_idx] = self._x.point.detach()
                log_probs[keep_idx] = self._x.log_prob.detach()
                keep_idx += 1

        samples, log_probs = samples[:keep_idx], log_probs[:keep_idx]
        if self.batched:  # chain-first, as run_chains and diagnostics expect
            return samples.transpose(0, 1), log_probs.transpose(0, 1)
        return samples, log_probs

    def state_dict(self) -> dict:
        """Serializable snapshot of sampler state for resuming a chain."""
        return {
            "x": {
                "point": self._x.point.detach().clone(),
                "log_prob": (
                    self._x.log_prob.detach().clone()
                    if self._x.log_prob is not None
                    else None
                ),
                "grad": (
                    self._x.grad.detach().clone()
                    if self._x.grad is not None
                    else None
                ),
            },
            "acceptance_ratio": float(self.acceptance_ratio),
            "step": int(self._step),
            "generator": self.generator.get_state(),
            "noisy": self.noisy,
            "reeval": self.reeval,
            "proposals": [p.state_dict() for p in self.proposals],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore sampler state captured by :meth:`state_dict`.

        Preserves the concrete type of ``self._x`` (e.g. ``HamiltonianSample``)
        so that subsequent proposals can populate algorithm-specific fields.
        """
        x_state = state["x"]
        new_sample = self._x.__class__(
            point=x_state["point"].clone(),
            log_prob=(
                x_state["log_prob"].clone() if x_state["log_prob"] is not None else None
            ),
            grad=x_state["grad"].clone() if x_state["grad"] is not None else None,
        )
        self._x = new_sample
        self.acceptance_ratio = float(state["acceptance_ratio"])
        self._step = int(state["step"])
        self.generator.set_state(state["generator"])
        self.noisy = bool(state["noisy"])
        self.reeval = int(state["reeval"])
        for proposal, p_state in zip(self.proposals, state["proposals"]):
            proposal.load_state_dict(p_state)
