"""Autograd helpers shared by gradient-based proposals (HMC, MALA, NUTS)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

import torch


def log_prob_and_grad(
    target: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``target`` at ``point`` and return ``(log_prob, grad)``.

    Builds a fresh autograd graph for each call so no graph state leaks across
    sampler iterations. Returns detached tensors. The caller need not manage
    ``requires_grad`` on the input.

    Parameters
    ----------
    target : Callable
        Target log-density function. Takes a 1-D tensor and returns a 0-D tensor.
    point : torch.Tensor
        1-D position at which to evaluate.

    Returns
    -------
    log_prob : torch.Tensor
        0-D detached tensor.
    grad : torch.Tensor
        1-D detached tensor with the same shape and dtype as ``point``.
    """
    with torch.enable_grad():
        leaf = point.detach().clone().requires_grad_(True)
        log_prob = target(leaf)
        if log_prob.ndim > 0:
            log_prob = log_prob.squeeze()
        (grad,) = torch.autograd.grad(log_prob, leaf, create_graph=False)
    return log_prob.detach(), grad.detach()
