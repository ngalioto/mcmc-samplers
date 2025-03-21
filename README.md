# MCMC Samplers
A Python package for performing MCMC sampling with PyTorch.

## Installation

To install this package, run the following command from the terminal:

```sh
pip install mcmc-samplers
```

## Framework

Each sampling algorithm is run using the `__call__` method inherited from the `Sampler` base class. The differences in implementation between the samplers are primarily determined by how the following two attributes are defined:

* `proposals`
* `acceptance_kernels`

Common proposals, acceptance kernels, and pairings of these two objects are described next.

### Proposals

One of the most important design choices in creating an MCMC sampler is how to propose samples. Proposals should be easy to sample from, efficient to evaluate, and resemble the target distribution as closely as possible. This package implements the following proposals:

* `GaussianRandomWalk`: A Gaussian random walk.
* `AdaptiveCovariance`: A Gaussian random walk where the covariance adapts to match the empirical covariance of the past states of the Markov chain.
* `ScaledCovariance`: A Gaussian random walk where the covariance is a scaled version of another (possibly time-varying) covariance.
* `HamiltonianDynamics`: A gradient-based proposal that uses Hamiltonian dynamics to propose samples from high-density regions.

### Acceptance kernels

The acceptance kernel in an MCMC scheme is used to guarantee that the Markov chain converges to the target distribution. The most common acceptance kernel comes from the Metropolis-Hastings algorithm, and most other acceptance kernels are based on the one used in that algorithm. Indeed, all of the samplers in this package use an acceptance kernel derived from the Metropolis-Hastings one.

### Samplers

Any MCMC sampler can be seen simply as a proposal paired with an acceptance kernel. This package implements the following common pairings:

* `MetropolisHastings`: Pairs the Metropolis-Hastings acceptance kernel with a Gaussian random walk proposal.
* `DelayedRejectionAdaptiveMetropolis`: Pairs the delayed rejection algorithm using two stages with the Gaussian random walk proposal using an adaptive covariance.
* `HamiltonianMonteCarlo`: Pairs the Metropolis-Hastings acceptance kernel with a Hamiltonian dynamics proposal.

## Example: Banana distribution

A walkthrough of part of the [banana](banana.py) example is given here to demonstrate how the package can be used.

First import the necessary packages.

```python
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from mcmc_samplers import *
```

Then define a function that evaluates the log probability of the (unnormalized) target distribution. In this example, a class is used.

```python
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
    ) -> torch.Tensor:
        
        x = torch.atleast_2d(x)
        y = torch.cat((x[:,0:1], x[:,1:2] + (x[:,0:1] + 1)**2), dim=1)
        return self.mvn.log_prob(y)
```

Inside the main function, instantiate the `Banana` class and specify sampling parameters. This example uses the DRAM algorithm, so the initial sample and initial covariance for the adaptive Gaussian random walk must be specified.

```python
target = Banana()

init_sample = torch.tensor([0.,-1.])
init_cov = torch.tensor([[1., 0.9], [0.9, 1.]])
```

Lastly, create the `Sampler` object and run for the desired number of iterations.

```python
dram = DelayedRejectionAdaptiveMetropolis(
        target = target.log_prob,
        x0 = init_sample,
        cov = init_cov
    )

num_samples = int(1e4)
samples, log_probs = dram(num_samples)
```

To visualize the results, the samples can be used to create a `SamplerVisualizer` object. This object can then be called to plot the sample chains and 1D and 2D histograms.
```python
import matplotlib.pyplot as plt

labels = ['$x_1$', '$x_2$']
visualizer = SampleVisualizer(samples)
visualizer.chains(labels=labels)
visualizer.triangular_hist(bins=50, labels=labels)
plt.show()
```

## Author information

Author: Nicholas Galioto  
Email: [ngalioto@umich.edu](mailto:ngalioto@umich.edu)  
License: GPL3  
Copyright 2025, Nicholas Galioto