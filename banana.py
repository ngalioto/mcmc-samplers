import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from base import Sample

import samplers

import matplotlib.pyplot as plt


class Banana:

    def __init__(
            self
    ):
        mean = torch.zeros(2)
        cov = torch.tensor([[1., 0.9], [0.9, 1.]])
        self.mvn = MultivariateNormal(mean, cov)

    def log_prob(
            self,
            x : torch.Tensor
    ):
        x = torch.atleast_2d(x)
        y = torch.cat((x[:,0:1], x[:,1:2] + (x[:,0:1] + 1)**2), dim=1)
        return self.mvn.log_prob(y)




if __name__ == "__main__":
    target = Banana()

    init_sample = torch.tensor([0.,-1.])
    init_cov = torch.tensor([[1., 0.9], [0.9, 1.]])

    num_samples = int(1e4)

    dram = samplers.Delayed_Rejection_Adaptive_Metropolis(
        target = target.log_prob,
        x0 = init_sample,
        cov = init_cov
    )

    samples, log_probs = dram(num_samples)
    print(dram.acceptance_ratio)
    plt.plot(log_probs)

    plt.figure()
    plt.plot(samples[:,0])
    plt.figure()
    plt.hist(samples[:,0])
    plt.figure()
    plt.hist(samples[:,1])
    plt.show()
