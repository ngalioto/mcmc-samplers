import torch
from torch.nn import Parameter
from mcmc_samplers import Sample
from typing import Union

"""
Defines the HamiltonianSample class as a child of the `Sample` base class.
"""

class HamiltonianSample(Sample):

    """
    A child of the Sample class for Hamiltonian Monte Carlo (HMC). Objects belonging to this class carry a position and momentum variable.

    Attributes
    ----------
    q : Sample
        The position variable. The point value is initialized as torch.nn.Parameter so that gradients with resepct to `q` can be computed with `backward`.
    p : torch.Tensor
        The momentum variable. Auxiliary variable in HMC.
    """

    def __init__(
            self,
            q = Union[Sample, torch.Tensor],
            p : torch.Tensor = None
    ):

        """
        HamiltonianSample constructor.

        Parameters
        ----------
        q : Union[Sample, torch.Tensor]
            The position of the Hamiltonian dynamics. Either a Sample or torch.Tensor can be passed.
        p : torch.Tensor, optional
            The momentum of the Hamiltonian dynamics.
        """
        
        if isinstance(q, Sample):
            self.q = Sample(Parameter(q.point))
            self.log_prob = q.log_prob
        elif isinstance(q, torch.Tensor):
            self.q = Sample(Parameter(q))
        else:
            raise ValueError("q must be of type either Sample or torch.Tensor")
        self.p = p

    @property
    def point(
            self
    ) -> torch.Tensor:

        """
        Getter for the `point` property.

        Returns
        ----------
        torch.Tensor
            The value of the position sample.
        """
        
        return self.q.point

    @point.setter
    def point(
            self,
            value : torch.Tensor
    ):

        """
        Setter for the `point` property.

        Parameters
        ----------
        value : torch.Tensor
            The position value of the Hamiltonian system.
        """
        
        self.q.point = value

    @property
    def log_prob(
            self
    ) -> torch.Tensor:

        """
        Getter for the `log_prob` property.

        Returns
        ----------
        torch.Tensor
            The log probability of a distribution of interested evaluated at the value of `self.point`.
        """
        
        return self.q.log_prob

    @log_prob.setter
    def log_prob(
            self,
            value : torch.Tensor
    ):

        """
        Setter for the `log_prob` property.

        Parameters
        ----------
        value : torch.Tensor
            The log probability of a distribution of interested evaluated at `self.point`.
        """
        
        self.q.log_prob = value
