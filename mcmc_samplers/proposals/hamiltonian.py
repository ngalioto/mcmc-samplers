import torch
from torch.nn import Parameter
from mcmc_samplers import Proposal, HamiltonianSample, Sample
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple
from collections.abc import Callable

"""
Defines the HamiltonianDynamics class as a child of the `Proposal` base class.
"""

class HamiltonianDynamics(Proposal):
    
    '''
    Proposal using Hamiltonian dynamics.

    Attributes
    ----------
    target : Callable[[torch.Tensor], [torch.Tensor]]
        Function that evaluates the log probability of the target distribution. In the Hamiltonian dyanmics, this represents the negative potential energy.
    dim : int
        Dimension of the sampler.
    step_size : float
        Step size to use to integrate the Hamiltonian dynamics.
    num_steps : int
        Number of steps to integrate the Hamiltonian dynamics.
    kinetic_energy : torch.Distribution.MultivariateNormal
        The negative log of this distribution represents the kinetic energy.
    '''
    
    def __init__(
            self,
            target : Callable[[torch.Tensor], [torch.Tensor]],
            dim : int,
            step_size : float = 1e-1,
            num_steps : int = 1
    ):

        """
        Description

        Parameters
        ----------
        target : Callable[[torch.Tensor], [torch.Tensor]]
            Function that evaluates the log probability of the target distribution. In the Hamiltonian dyanmics, this represents the negative potential energy.
        dim : int
            Dimension of the sampler.
        step_size : float
            Step size to use to integrate the Hamiltonian dynamics. Default is 1e-1.
        num_steps : int
            Number of steps to integrate the Hamiltonian dynamics. Default is 1.
        """

        self.target = target
        self.dim = dim
        self.step_size = step_size
        self.num_steps = num_steps

        self.kinetic_energy = MultivariateNormal(
            loc = torch.zeros(self.dim),
            covariance_matrix = torch.eye(self.dim)
        )

    @property
    def is_symmetric(
            self
    ) -> bool:

        """
        Getter for property `is_symmetric`. 

        Returns
        ----------
        bool
            The Hamiltonian dynamics proposal is not symmetric, so value is always False.
        """
        
        return False

    def _leapfrog(
            self,
            y : HamiltonianSample
    ) -> HamiltonianSample:

        """
        Leapfrog integrator.

        Parameters
        ----------
        y : HamiltonianSample
            The current position and momentum of the sampler stored as a HamiltonianSample object.

        Returns
        ----------
        HamiltonianSample
            The position and momentum approximately integrated `self.step_size * self.num_steps` forward in time and stored in a HamiltonianSample object.
        """

        q = Sample(y.point.detach().clone())
        p = y.p.clone()

        # Integrate for `self.num_steps` steps
        p = p + (self.step_size / 2) * y.point.grad
        for ii in range(self.num_steps):
            q.point = Parameter(q.point + self.step_size * p)
            q.log_prob = self.target(q.point)
            q.log_prob.backward()
            if ii != self.num_steps - 1:
                p = p + self.step_size * q.point.grad
        p = p + (self.step_size / 2) * q.point.grad
        
        return HamiltonianSample(q, p)

    def propose(
            self,
            y : Tuple[HamiltonianSample, ...]
    ) -> HamiltonianSample:

        """
        Proposes a sample using Hamiltonian dynamics.

        Parameters
        ----------
        y : Tuple[HamiltonianSample, ...]
            List of current and proposed HamiltonianSample objects

        Returns
        ----------
        HamiltonianSample
            Proposed sample consisting of a position and momentum variable.
        """

        y[0].p = self.kinetic_energy.sample()
        return self._leapfrog(y[0])

    def log_prob(
            self,
            y : Tuple[HamiltonianSample, ...] = None
    ) -> torch.Tensor:

        """
        Evaluates the log probability of the proposal kernel.

        Parameters
        ----------
        y : Tuple[HamiltonianSample, ...]
            List of current and proposed HamiltonianSample objects.

        Returns
        ----------
        torch.Tensor
            Log probability density of the proposal. Equivalent to the negative kinetic energy function.
        """

        return self.kinetic_energy.log_prob(y[-1].p)
