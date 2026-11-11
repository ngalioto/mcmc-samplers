"""Sampling constrained parameters through an unconstrained transform."""

import torch
from torch.distributions import biject_to, constraints

from mcmc_samplers import MetropolisHastings, transformed_target


def test_exponential_through_log_transform():
    # x ~ Exponential(1) on x > 0; sample z = log(x) instead.
    def target(x):
        return -x.sum()

    positive = biject_to(constraints.positive)
    sampler = MetropolisHastings(
        transformed_target(target, positive),
        x0=positive.inv(torch.ones(1)),
        cov=torch.eye(1),
        generator=torch.Generator().manual_seed(0),
    )
    z, _ = sampler(20_000, burn_in=1000, show_progress=False)
    x = positive(z)
    assert (x > 0).all()
    assert abs(x.mean().item() - 1.0) < 0.1
    assert abs(x.var().item() - 1.0) < 0.25


def test_jacobian_is_summed_over_coordinates():
    def target(x):
        return torch.zeros(())

    positive = biject_to(constraints.positive)
    z = torch.tensor([0.5, -1.0, 2.0])
    # log|det J| of exp is sum(z)
    assert torch.allclose(transformed_target(target, positive)(z), z.sum())
