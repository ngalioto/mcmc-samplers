"""Compare samplers by effective samples per second and per target evaluation.

Run from the repo root, with this repo installed (``pip install -e .``):

    python benchmarks/benchmark.py

For each target and sampler it reports the smallest per-coordinate ESS,
divided by wall-clock time and by the number of target evaluations (a
log-density-plus-gradient evaluation counts as one). The second number is the
one that matters when the target is expensive, e.g. a filter likelihood.
Settings are reasonable hand-picked defaults, not tuned per target.
"""

import time

import torch

from mcmc_samplers import (
    NUTS,
    AdaptiveMetropolis,
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    MetropolisAdjustedLangevin,
    MetropolisHastings,
    diagnostics,
)

N = 4000
BURN_IN = 1000


class Counted:
    """Wrap a log-density and count how often it is called."""

    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)


def correlated_gaussian(dim=10):
    sd = torch.linspace(0.5, 3.0, dim)
    idx = torch.arange(dim)
    corr = 0.9 ** (idx[:, None] - idx[None, :]).abs()
    cov = sd[:, None] * corr * sd[None, :]
    dist = torch.distributions.MultivariateNormal(torch.zeros(dim), cov)
    return dist.log_prob, torch.zeros(dim)


def banana():
    dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    )

    def log_prob(x):
        y = torch.stack([x[0], x[1] + (x[0] + 1.0) ** 2])
        return dist.log_prob(y)

    return log_prob, torch.tensor([0.0, -1.0])


def samplers(target, x0, gen):
    dim = x0.shape[0]
    cov = 0.1 * torch.eye(dim)
    return {
        "MH": MetropolisHastings(target, x0, cov=cov, generator=gen),
        "AM": AdaptiveMetropolis(target, x0, cov=cov, n0=200, generator=gen),
        "DRAM": DelayedRejectionAdaptiveMetropolis(target, x0, cov=cov, n0=200, generator=gen),
        "MALA": MetropolisAdjustedLangevin(target, x0, step_size=0.3, generator=gen),
        "HMC": HamiltonianMonteCarlo(target, x0, step_size=0.15, num_steps=10, generator=gen),
        "NUTS": NUTS(target, x0, warmup=BURN_IN, generator=gen),
    }


def run(target_name, make_target):
    print(f"\n### {target_name}\n")
    print("| Sampler | accept | min ESS | ESS / s | ESS / 1000 evals |")
    print("|---|---|---|---|---|")
    for name in ["MH", "AM", "DRAM", "MALA", "HMC", "NUTS"]:
        log_prob, x0 = make_target()
        counted = Counted(log_prob)
        sampler = samplers(counted, x0, torch.Generator().manual_seed(0))[name]
        counted.calls = 0  # ignore evaluations made while constructing
        start = time.perf_counter()
        draws, _ = sampler(N, burn_in=BURN_IN, show_progress=False)
        seconds = time.perf_counter() - start
        ess = float(diagnostics.effective_sample_size(draws).min())
        print(
            f"| {name} | {sampler.acceptance_ratio:.2f} | {ess:.0f} | "
            f"{ess / seconds:.0f} | {1000 * ess / counted.calls:.1f} |",
            flush=True,
        )


if __name__ == "__main__":
    print(f"{N} steps per sampler ({BURN_IN} burn-in).")
    run("10-d correlated Gaussian", correlated_gaussian)
    run("2-d banana", banana)
