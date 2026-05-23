"""Abstract base class for MCMC proposal distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union

import torch

from ..samples.sample import Sample


class Proposal(ABC):
    """Abstract base class for proposal distributions.

    A proposal must implement three things:

    - ``is_symmetric``: whether ``log_prob(y | x) == log_prob(x | y)``.
    - ``propose``: draw a proposed :class:`~mcmc_samplers.Sample` (or list of
      Samples) conditional on the current chain state.
    - ``log_prob``: evaluate the (un-normalized) proposal log-density at the
      proposed sample given the current state.

    For correctness across the sampler stack, ``propose`` and ``log_prob`` both
    take ``y``: a list whose entries represent the current state followed by all
    previous-stage proposals in this iteration (delayed rejection). The
    convention is ``y[0]`` is the current chain state and ``y[-1]`` is the most
    recent proposal.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_symmetric(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def propose(
        self,
        y: Sequence[Sample],
        N: int = 1,
        generator: torch.Generator | None = None,
    ) -> Union[Sample, list]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, y: Sequence[Sample]) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Return a serializable snapshot of mutable proposal state.

        Subclasses override to include adaptation state, scale parameters, etc.
        Tensors are cloned and detached.
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore mutable state from a snapshot produced by :meth:`state_dict`."""
        return None
