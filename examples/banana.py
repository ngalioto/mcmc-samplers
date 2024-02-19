import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from mcmc_samplers import *

import matplotlib.pyplot as plt

"""
Toy example to demonstrate the functionality of the DRAM and HMC samplers.
"""

class Banana:

    """
    2D dimensional target distribution with a banana-shaped joint distribution.

    Attributes
        ----------
        mvn : MultivariateNormal
            Multivariate normal distribution that will be used to evaluate the log probability.
    """

    def __init__(
            self
    ):

        """
        Banana constructor.
        """
        
        mean = torch.zeros(2)
        cov = torch.tensor([[1., 0.9], [0.9, 1.]])
        self.mvn = MultivariateNormal(mean, cov)

    def log_prob(
            self,
            x : torch.Tensor
    ) -> torch.Tensor:

        """
        Evaluates the log probability of the banana distribution.

        Parameters
        ----------
        x : torch.Tensor
            Point at which to evaluate the log probability density.

        Returns
        ----------
        torch.Tensor
            The log probability density of the banana distribution at the point `x`.
        """
        
        x = torch.atleast_2d(x)
        y = torch.cat((x[:,0:1], x[:,1:2] + (x[:,0:1] + 1)**2), dim=1)
        return self.mvn.log_prob(y)




if __name__ == "__main__":

    torch.manual_seed(0) # for reproducibility
    
    target = Banana()

    init_sample = torch.tensor([0.,-1.])
    init_cov = torch.tensor([[1., 0.9], [0.9, 1.]])

    num_samples = int(1e4)


    dram = DelayedRejectionAdaptiveMetropolis(
        target = target.log_prob,
        x0 = init_sample,
        cov = init_cov
    )
    
    hmc = HamiltonianMonteCarlo(
        target = target.log_prob,
        x0 = torch.nn.Parameter(init_sample),
        step_size = 0.2,
        num_steps = 3
    )

    samplers = [dram, hmc]
    titles = ['DRAM', 'HMC']
    labels = ['$x_1$', '$x_2$']
    visualizers = [None, None]

    for ii, sampler in enumerate(samplers):
        samples, log_probs = sampler(num_samples)
        print(f'{titles[ii]} acceptence rate: {100*sampler.acceptance_ratio:.2f}%')
    
        visualizers[ii] = SampleVisualizer(samples)
        visualizers[ii].triangular_hist(bins=50,
                                   labels=labels,
                                   titles=[titles[ii]])
        visualizers[ii].chains(labels=labels,
                               titles=[titles[ii]])

    plt.show()
