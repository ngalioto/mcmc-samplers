"""Helpers for running multiple MCMC chains."""

from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

import torch

from ..samplers.sampler import Sampler


def run_chains(
    sampler_factory: Callable[[int], Sampler],
    num_chains: int,
    N: int,
    burn_in: int = 0,
    thin: int = 1,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run several independent chains sequentially and stack the results.

    Parameters
    ----------
    sampler_factory : Callable[[int], Sampler]
        Builds one sampler given a chain index. Use the index to differ
        starting points and / or RNG seeds across chains. For example::

            def factory(chain_idx):
                gen = torch.Generator().manual_seed(chain_idx)
                return MetropolisHastings(target, x0, cov=cov, generator=gen)

    num_chains : int
        How many independent chains to run.
    N, burn_in, thin, show_progress :
        Forwarded to each sampler's ``__call__``.

    Returns
    -------
    chains : torch.Tensor
        Stacked samples of shape ``(num_chains, kept, dim)`` where
        ``kept = (N - burn_in + thin - 1) // thin``.
    log_probs : torch.Tensor
        Stacked log-densities of shape ``(num_chains, kept)``.

    Notes
    -----
    Chains run sequentially. For true parallelism use
    ``torch.multiprocessing`` to fan out and call this function per worker,
    then ``torch.cat`` the results.
    """
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}")

    all_samples = []
    all_log_probs = []
    for chain_idx in range(num_chains):
        sampler = sampler_factory(chain_idx)
        samples, log_probs = sampler(
            N=N, burn_in=burn_in, thin=thin, show_progress=show_progress
        )
        all_samples.append(samples)
        all_log_probs.append(log_probs)

    return torch.stack(all_samples, dim=0), torch.stack(all_log_probs, dim=0)
