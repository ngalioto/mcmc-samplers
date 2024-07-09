import torch
from torch.nn import Parameter
from mcmc_samplers import DelayedRejection, HamiltonianDynamics, HamiltonianSample, Sample
from typing import Union
from collections.abc import Callable

"""
Defines the HamiltonianMonteCarlo class as a child of the `DelayedRejection` base class
"""

class HamiltonianMonteCarlo(DelayedRejection):

    """
    Implementation of the Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) algorithm.

    Citation:
        Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of Markov chain Monte Carlo 2.11 (2011): 2.

    Attributes
    ----------
    target : Callable[[torch.Tensor], [torch.Tensor]],
        Function that evaluates the log probability of the target distribution.
    x0 : HamiltonianSample
        Initial point of the Markov chain.
    dim : int
        Dimension of the sampler.
    proposals : Tuple[Proposal]
        HamiltonianDynamics proposal.
    acceptance_kernels : Tuple[Callable[[Tuple[Sample, ...]],[torch.Tensor]]]
        The Metropolis-Hastings acceptance kernel. In this case, it is the ratio of Hamiltonians of the current and proposed points.
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            step_size : float = 1e-1,
            num_steps : int = 1,
            noisy : bool = False,
            reeval : int = 200
    ):

        """
        HamiltonianMonteCarlo constructor.

        Parameters
        ----------
        target : Callable[[torch.Tensor], [torch.Tensor]]
        Function that evaluates the log probability density of a target distribution
        x0 : Union[Sample, torch.Tensor]
            Initial point of the Markov chain for the position variables.
        step_size : float
            Step size to integrate the Hamiltonian dynamics. Default is 1e-1.
        num_steps : int
            Number of steps to integrate the Hamiltonian dynamics. Default is 1.
        """

        self.dim = len(x0)
        super().__init__(
            target = target,
            x0 = x0,
            proposals = [
                HamiltonianDynamics(
                    target,
                    self.dim,
                    step_size,
                    num_steps
                )
            ],
            noisy = noisy,
            reeval = reeval
        )


    @property
    def x(
            self
    ) -> HamiltonianSample:

        """
        Getter for the `x` property.

        Returns
        ----------
        HamiltonianSample
            Current position and momentum of the sampler
        """
        
        return self._x

    @x.setter
    def x(
            self,
            value : Union[torch.Tensor, Sample]
    ):

        """
        Setter for the `x` property. Initializes the position variable as a torch.nn.Parameter. Then it evaluates the `self.target` and calls `backward` to compute the gradient.

        Parameters
        ----------
        value : Union[torch.Tensor, Sample]
           Either a torch.Tensor or Sample representing the position variable of the Hamiltonian system.
        """

        if isinstance(value, HamiltonianSample):
            self._x = HamiltonianSample(value.q, value.p)
        else:
            self._x = HamiltonianSample(value)

        # compute gradient    
        self._x.log_prob = self.target(self._x.point)
        self._x.log_prob.backward()
        
