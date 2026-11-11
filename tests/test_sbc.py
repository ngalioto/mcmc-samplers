"""Simulation-based calibration (Talts et al. 2018).

If a sampler targets the posterior correctly, then for theta drawn from the
prior and data simulated from theta, the rank of theta among (roughly
independent) posterior draws is uniform. We check the two moments of the
normalized rank u = (rank + 0.5) / (L + 1): its mean (0.5 unless the sampler
is shifted) and its variance (1/12 unless the sampler is too wide or too
narrow). These catch errors that mean/covariance checks on a known target miss.

Model: theta ~ N(0, 1), y_i ~ N(theta, 1) for i = 1..5.

The one-chain-at-a-time checks take a few minutes, so they are marked slow
(run with ``pytest -m slow``). The batched checks run every dataset as its own
chain in one batched sampler and take about a second.
"""

import pytest
import torch

from mcmc_samplers import (
    AdaptiveMetropolis,
    DelayedRejectionAdaptiveMetropolis,
    MetropolisAdjustedLangevin,
    MetropolisHastings,
    PreconditionedCrankNicolsonSampler,
)

NUM_REPS = 400  # simulated datasets
NUM_DRAWS = 19  # posterior draws per dataset (L), so ranks are 0..19
THIN = 4
BURN_IN = 100
Z_MAX = 3.5  # each z-statistic is ~N(0, 1) for a correct sampler


def _make_sampler(name, y, generator):
    def log_lik(theta):
        return -0.5 * ((y - theta) ** 2).sum()

    def log_post(theta):
        return log_lik(theta) - 0.5 * (theta**2).sum()

    x0 = torch.zeros(1)
    step = 0.5 * torch.eye(1)
    if name == "MH":
        return MetropolisHastings(log_post, x0, cov=step, generator=generator)
    if name == "DRAM":
        return DelayedRejectionAdaptiveMetropolis(
            log_post, x0, cov=step, n0=50, generator=generator
        )
    if name == "MALA":
        return MetropolisAdjustedLangevin(log_post, x0, step_size=0.6, generator=generator)
    if name == "pCN":
        return PreconditionedCrankNicolsonSampler(
            log_lik, x0, prior_cov=torch.eye(1), beta=0.5, generator=generator
        )
    raise ValueError(name)


def sbc_z_scores(make_sampler):
    """Return (z_mean, z_var) of the normalized SBC ranks."""
    gen = torch.Generator().manual_seed(1)
    ranks = []
    for _ in range(NUM_REPS):
        theta = torch.randn(1, generator=gen)
        y = theta + torch.randn(5, generator=gen)
        seed = int(torch.randint(0, 2**31, (1,), generator=gen))
        sampler = make_sampler(y, torch.Generator().manual_seed(seed))
        draws, _ = sampler(
            BURN_IN + NUM_DRAWS * THIN, burn_in=BURN_IN, thin=THIN, show_progress=False
        )
        ranks.append(int((draws[:, 0] < theta).sum()))

    return _z_scores(torch.tensor(ranks))


def _z_scores(ranks):
    u = (ranks.to(torch.float64) + 0.5) / (NUM_DRAWS + 1)
    z_mean = (u.mean() - 0.5) / (1 / 12 / NUM_REPS) ** 0.5
    z_var = (u.var() - 1 / 12) / ((1 / 80 - 1 / 144) / NUM_REPS) ** 0.5
    return float(z_mean), float(z_var)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["MH", "DRAM", "MALA", "pCN"])
def test_sbc_ranks_are_uniform(name):
    z_mean, z_var = sbc_z_scores(lambda y, gen: _make_sampler(name, y, gen))
    assert abs(z_mean) < Z_MAX, f"{name} looks biased (z = {z_mean:.2f})"
    assert abs(z_var) < Z_MAX, f"{name} looks too wide/narrow (z = {z_var:.2f})"


BATCHED = {
    "MH": MetropolisHastings,
    "AM": AdaptiveMetropolis,
    "DRAM": DelayedRejectionAdaptiveMetropolis,
}


@pytest.mark.parametrize("name", BATCHED)
def test_sbc_batched(name):
    """Same check, with dataset i run as chain i of one batched sampler."""
    gen = torch.Generator().manual_seed(1)
    theta = torch.randn(NUM_REPS, 1, generator=gen)
    y = theta + torch.randn(NUM_REPS, 5, generator=gen)

    def log_post(t):  # t: (NUM_REPS, 1), one dataset per chain
        return -0.5 * ((y - t) ** 2).sum(-1) - 0.5 * (t**2).sum(-1)

    sampler = BATCHED[name](
        log_post, torch.zeros(NUM_REPS, 1), cov=0.5 * torch.eye(1),
        generator=torch.Generator().manual_seed(2), batched=True,
    )
    draws, _ = sampler(BURN_IN + NUM_DRAWS * THIN, burn_in=BURN_IN, thin=THIN, show_progress=False)
    ranks = (draws[..., 0] < theta).sum(dim=1)
    z_mean, z_var = _z_scores(ranks)
    assert abs(z_mean) < Z_MAX, f"{name} looks biased (z = {z_mean:.2f})"
    assert abs(z_var) < Z_MAX, f"{name} looks too wide/narrow (z = {z_var:.2f})"
