import torch
from mcmc_samplers import Proposal, Sample
from abc import ABC
from tqdm import tqdm
from collections.abc import Callable
from typing import Union, Tuple

"""
Defines the Sampler abstract base class. Sampler objects are used to execute MCMC algorithms.
"""

class Sampler(ABC):
    
    """
    Abstract base class of the Sampler class.

    Attributes
    ----------
    target : Callable[[torch.Tensor], [torch.Tensor]]
        Function that evaluates the log probability density of a target distribution.
    x : Sample
        Sample representing the current position of the Markov chain
    proposals : Tuple[Proposal, ...]
        List of proposals.
    acceptance_kernels : Tuple[Callable[[Tuple[Sample, ...]],[torch.Tensor]], ...]
        List of acceptance kernels. 
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            proposals : Tuple[Proposal, ...],
            acceptance_kernels : Tuple[Callable[[Sample, Tuple[Sample, ...]],torch.Tensor], ...],
            noisy : bool = False,
            reeval : int = 200
    ):
        
        """
        Sampler constructor.

        Parameters
        ----------
        target : Callable[[torch.Tensor], [torch.Tensor]]
            evaluates the log probability of the target distribution.
        x0 : Union[Sample, torch.Tensor]
            point at which to initialize the Markov chain
        proposals : Tuple[Proposal, ...]
            list of proposals. 
        acceptance_kernels : Tuple[Callable[[Sample, Tuple[Sample, ...]],[torch.Tensor]], ...]
            list of acceptance kernels. Each acceptance kernel evaluates the log acceptance probability of a proposed sample given the current position of the Markov chain (and possibly given rejected proposals at earlier stages)
        """

        self.acceptance_ratio = 0
        self.target = target
        self.x = torch.atleast_2d(x0)
        self.proposals = proposals
        self.acceptance_kernels = acceptance_kernels
        self.noisy = noisy
        self.reeval = reeval

    @property
    def x(
            self
    ) -> Sample:
        
        """
        Getter for the property `x`

        Returns
        ----------
        Sample
            the current position of the Markov chain.
        """
        
        return self._x

    @x.setter
    def x(
            self,
            value : Union[Sample, torch.Tensor]
    ):
        
        """
        Setter for the property `x`.

        Parameters
        ----------
        value : Union[Sample, torch.Tensor]
            Markov chain state at which to position the sampler.
        """
        
        if isinstance(value, Sample):
            self._x = value
            if value.log_prob is None:
                self._x.log_prob = self.target(value.point)
        elif isinstance(value, torch.Tensor):
            self._x = Sample(value, self.target(value))
        else:
            raise ValueError(f'x must be of type either Sample or torch.Tensor')

    def _sample(
            self
    ) -> int:
        
        """
        Advances the Markov chain forward one step

        Returns
        ----------
        int
            1 if a proposed sample is accepted. In this case the sampler moves its position to the accepted point
            0 if all proposed samples are rejected. In this case, the sampler remains in its current state.
        """
        
        y = [self.x]
        for proposal, acceptance_prob in zip(self.proposals, self.acceptance_kernels):
            y.append(proposal.propose(y))
            alpha = acceptance_prob(y)
            if alpha > torch.log(torch.rand(1)): # accept
                self.x = y[-1]
                return 1
        return 0 # proposal rejected at every stage

    def __call__(
            self,
            N : int = 1,
            show_progress : bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Runs the sampler for as long as specified.

        Parameters
        ----------
        N : int
            Number of steps to advance the sampler. Default is 1

        Returns
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            The first return object is a tensor representing the Markov chain over `N` iterations
            The second return object is a tensor holding the `target` log probability values of each state in the Markov chain.
        """

        rejection_streak = 0

        samples = torch.zeros(N, self.x.point.shape[1])
        log_probs = torch.zeros(N)
        for ii in tqdm(range(N)) if show_progress else range(N):
            accept = self._sample()
            self.acceptance_ratio = (ii * self.acceptance_ratio + accept) / (ii+1)
            samples[ii] = self.x.point.detach()

            if self.noisy:
                rejection_streak = 0 if accept else rejection_streak + 1
                if rejection_streak == self.reeval:
                    self.x.log_prob = self.target(self.x.point)
                    rejection_streak = 0

            log_probs[ii] = self.x.log_prob
            
        return samples, log_probs
