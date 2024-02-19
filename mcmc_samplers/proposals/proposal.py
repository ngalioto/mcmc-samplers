import torch
from mcmc_samplers import Sample
from abc import ABC, abstractmethod
from typing import Tuple

"""
Defines the Proposal base class. Proposal objects propose states to which to move an MCMC sampler. Proposal objects are also able to evaluate the log probability of their underlying proposal distribution.
"""

class Proposal(ABC):
    
    """
    Abstract base class of the Proposal class.

    Attributes
    ----------
    is_symmetric : bool
        True if the proposal is symmetric and false otherwise.
    """

    @abstractmethod
    def __init__(
            self
    ):
        
        """
        Proposal constructor.
        """
        
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_symmetric(
            self
    ) -> bool:

        """
        Getter of the property `is_symmetric``.
        """
        
        raise NotImplementedError()

    @abstractmethod
    def propose(
            self,
            y : Tuple[Sample, ...]
    ) -> Sample:
        
        """
        Generates a sample from the proposal distribution.
        """
        
        raise NotImplementedError()

    @abstractmethod
    def log_prob(
            self,
            y : Tuple[Sample, ...]
    ) -> torch.Tensor:
        
        """
        Evaluates the log of unnormalized probability density function
        """
        
        raise NotImplementedError()
