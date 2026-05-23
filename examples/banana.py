"""Toy example exercising DRAM and HMC on the classic banana distribution."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from mcmc_samplers import (
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    SampleVisualizer,
)


class Banana:
    """A 2-D banana-shaped joint density."""

    def __init__(self):
        mean = torch.zeros(2)
        cov = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
        self.mvn = MultivariateNormal(mean, cov)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atleast_2d(x)
        y = torch.cat((x[:, 0:1], x[:, 1:2] + (x[:, 0:1] + 1.0) ** 2), dim=1)
        return self.mvn.log_prob(y).squeeze()


if __name__ == "__main__":
    target = Banana()
    init_sample = torch.tensor([0.0, -1.0])
    init_cov = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    num_samples = 10_000

    g = torch.Generator().manual_seed(0)
    dram = DelayedRejectionAdaptiveMetropolis(
        target=target.log_prob,
        x0=init_sample,
        cov=init_cov,
        generator=g,
    )

    g2 = torch.Generator().manual_seed(0)
    hmc = HamiltonianMonteCarlo(
        target=target.log_prob,
        x0=init_sample,
        step_size=0.2,
        num_steps=3,
        generator=g2,
    )

    samplers = [dram, hmc]
    titles = ["DRAM", "HMC"]
    labels = ["$x_1$", "$x_2$"]

    for sampler, title in zip(samplers, titles):
        samples, _ = sampler(num_samples, burn_in=num_samples // 5)
        print(f"{title} acceptance rate: {100.0 * sampler.acceptance_ratio:.2f}%")
        visualizer = SampleVisualizer(samples)
        visualizer.triangular_hist(bins=50, labels=labels, titles=[title])
        visualizer.chains(labels=labels, titles=[title])

    plt.show()
