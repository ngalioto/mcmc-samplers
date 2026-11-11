"""Sampling constrained parameters in an unconstrained space."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.distributions import Transform


def transformed_target(
    target: Callable[[torch.Tensor], torch.Tensor],
    transform: Transform,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Rewrite a constrained target as a log-density on unconstrained space.

    With ``x = transform(z)``, the returned function is

    .. math:: \\log p(z) = \\log \\pi(x) + \\log |\\det J(z)|,

    so any sampler can run on ``z`` without proposing invalid values. Map the
    samples back with ``transform(samples)``.

    Parameters
    ----------
    target : Callable
        Log-density of the constrained parameter ``x``.
    transform : torch.distributions.Transform
        Bijection from unconstrained ``z`` to ``x``. ``torch.distributions``
        provides one for each constraint, e.g.
        ``biject_to(constraints.positive)`` for positive parameters. Use
        ``CatTransform`` to give different coordinates different constraints.

    Example
    -------
    >>> from torch.distributions import biject_to, constraints
    >>> positive = biject_to(constraints.positive)        # z -> exp(z)
    >>> sampler = MetropolisHastings(transformed_target(target, positive),
    ...                              x0=positive.inv(x0), cov=cov)
    >>> z, _ = sampler(N=5000)
    >>> x = positive(z)
    """

    def log_density(z: torch.Tensor) -> torch.Tensor:
        x = transform(z)
        log_det = transform.log_abs_det_jacobian(z, x)
        if transform.domain.event_dim == 0:
            log_det = log_det.sum(-1)  # elementwise transform: sum over coordinates
        return target(x) + log_det

    return log_density
