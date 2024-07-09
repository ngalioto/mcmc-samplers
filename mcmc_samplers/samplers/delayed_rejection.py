import torch
from mcmc_samplers import *
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, Tuple

"""
Defines the `DelayedRejection` abstract base class and all MCMC samplers that are derived from it.
"""

class DelayedRejection(Sampler, ABC):
    
    """
    The `DelayedRejection` abstract base class. It is a child class of the `Sampler` class. This class executes the multi-tiered proposal algorithm known as ``delayed rejection.''

    Citation:
        Green, Peter J., and Antonietta Mira. ``Delayed rejection in reversible jump Metropolis-Hastings.'' Biometrika 88.4 (2001): 1035-1053.

    Attributes
    ----------
    Carries the same attributes as the parent class `Sampler`.
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            proposals : Tuple[Proposal, ...],
            num_stages : int = 1,
            noisy : bool = False,
            reeval : int = 200
    ):
        
        """
        `DelayedRejection` constructor. Constructs the acceptance_kernels according to the delayed rejection algorithm.

        Parameters
        ----------
        target : Callable[[torch.Tensor], [torch.Tensor]]
            evaluates the log probability of the target distribution.
        x0 : Union[Sample, torch.Tensor]
            point at which to initialize the Markov chain
        proposals : Tuple[Proposal, ...]
            list of proposals.
        num_stages : int
            Number of proposal stages/tiers. Default is 1
        """
        
        super().__init__(
            target = target,
            x0 = x0,
            proposals = proposals,
            acceptance_kernels = [lambda y, stage=ii : self._metropolis_hastings(y, stage) for ii in range(1, num_stages+1)],
            noisy = noisy,
            reeval = reeval
        )

        self._denom = None
        self._alpha = None

    def _metropolis_hastings(
            self,
            y : Tuple[Sample, ...],
            stage : int = 1
    ) -> torch.Tensor:
        
        """
        Evaluates the log acceptance probability of a proposed sample at a given stage level.

        Parameters
        ----------
        y : Tuple[Sample, ...]
           List containing samples representing the current state of the sampler, the proposed sample, and all rejected proposals, if any.  The ordering of the list is as follows: [x, y_1, ... , y_stage]
        stage : int
            The stage/tier of the proposal. Default is 1 representing the standard Metropolis-Hastings acceptance probability

        Returns
        ----------
        int
            the log acceptance probability
        """

        assert len(y) == stage + 1
        assert self._denom is not None
        
        if stage > 1:
            assert self._alpha is not None
        
        if y[-1].log_prob is None:
            y[-1].log_prob = self.target(y[-1].point)

        denom = self._denom.clone()
        if stage > 1:
            denom += torch.log(1 - torch.exp(self._alpha))
            if self.proposals[stage-2].is_symmetric:
                denom += self.proposals[stage-2].log_prob(y[:-1])

        self._denom = y[-1].log_prob.clone()
        for ii in range(1,stage):
            self.acceptance_kernels[ii-1](y[:-(ii+2):-1])

        numer = self._denom.clone()
        if stage > 1:
            numer += torch.log(1 - torch.exp(self._alpha))
            if self.proposals[stage-2].is_symmetric:
                numer += self.proposals[stage-2].log_prob(y[:0:-1])

        if not self.proposals[stage-1].is_symmetric:
            numer += self.proposals[stage-1].log_prob(y[::-1])
            denom += self.proposals[stage-1].log_prob(y)

        self._denom = denom
        self._alpha = numer - self._denom
        if torch.isnan(self._alpha):
            self._alpha = torch.tensor([-torch.inf])
        self._alpha = min(torch.zeros(1), self._alpha)
        return self._alpha

    def _sample(
            self
    ) -> int:
        
        """
        Advances the Markov chain one step. 

        Returns
        ----------
        int
            1 if a proposed sample is accepted and 0 otherwise.
        """
        
        self._denom = self.x.log_prob
        return super()._sample()

    
# can make this a child of adaptive metropolis
class DelayedRejectionAdaptiveMetropolis(DelayedRejection):

    """
    Implementation of the delayed rejection adaptive Metropolis (DRAM) sampler. This is a child class of the `DelayedRejection`

    Citation:
        Haario, Heikki, et al. "DRAM: efficient adaptive MCMC." Statistics and computing 16 (2006): 339-354.

    Attributes
    ----------
    The attributes are the same as those in the `DelayedRejection` parent class.    
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            cov : torch.Tensor = None,
            sd : float = None,
            gamma : float = 1e-2,
            n0 : int = 200,
            eps : float = 1e-8,
            noisy : bool = False,
            reeval : int = 200
    ):
        
        """
        DelayedRejectionAdaptiveMetropolis constructor.

        target : Callable[[torch.Tensor], [torch.Tensor]]
            function evaluating the log probability of the target distribution
        x0 : Union[Sample, torch.Tensor]
            initial state of the Markov chain
        cov : torch.Tensor, optional
            Tensor representing the proposal covariance
        sd : float, optional
            Scaling factor applied to the proposal covariance. By default, the value is set to be $2.4^2 / d$, where $d$ is the dimension of the proposal
        gamma : float
            Scaling factor applied to the first-stage proposal covariance to yield the second-stage proposal covariance. The default is 1e-2
        n0 : int
            Number of iterations to run the sampler before using the empirical sample covariance as the first-stage proposal covariance. Default is 200
        eps : float
            Small positive value added to the diagonal of the proposal covariance during adaptation to encourage positive-definitess. Default is 1e-8
        """
        
        proposals = [AdaptiveCovariance(cov, sd, n0, eps)]
        proposals.append(ScaledCovariance(proposals[0], gamma))
        super().__init__(
            target = target,
            x0 = x0,
            proposals = proposals,
            num_stages = 2,
            noisy = noisy,
            reeval = reeval
        )

    def _sample(
            self
    ) -> int:
        
        """
        Advances the Markov chain one step. First, this method adapts the proposal covariance with the adaptive Metropolis (AM) algorithm. Then it samples with the delayed rejection (DR) algorithm.

        Returns
        ----------
        int
            1 if a proposed sample is accepted and 0 otherwise.
        """
        
        self.proposals[0]._adapt(self.x.point.squeeze())
        return super()._sample()
    

class AdaptiveMetropolis(DelayedRejection):

    """
    Implementation of the adaptive Metropolis (AM) sampler. This is a child class of the `DelayedRejection`

    Attributes
    ----------
    The attributes are the same as those in the `DelayedRejection` parent class.    
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            cov : torch.Tensor = None,
            sd : float = None,
            n0 : int = 200,
            eps : float = 1e-8,
            noisy : bool = False,
            reeval : int = 200
    ):
        
        """
        AdaptiveMetropolis constructor.

        target : Callable[[torch.Tensor], [torch.Tensor]]
            function evaluating the log probability of the target distribution
        x0 : Union[Sample, torch.Tensor]
            initial state of the Markov chain
        cov : torch.Tensor, optional
            Tensor representing the proposal covariance
        sd : float, optional
            Scaling factor applied to the proposal covariance. By default, the value is set to be $2.4^2 / d$, where $d$ is the dimension of the proposal
        n0 : int
            Number of iterations to run the sampler before using the empirical sample covariance as the first-stage proposal covariance. Default is 200
        eps : float
            Small positive value added to the diagonal of the proposal covariance during adaptation to encourage positive-definitess. Default is 1e-8
        """
        
        proposals = [AdaptiveCovariance(cov, sd, n0, eps)]
        super().__init__(
            target = target,
            x0 = x0,
            proposals = proposals,
            num_stages = 1,
            noisy = noisy,
            reeval = reeval
        )

    def _sample(
            self
    ) -> int:
        
        """
        Advances the Markov chain one step. First, this method adapts the proposal covariance with the adaptive Metropolis (AM) algorithm. Then it samples with the delayed rejection (DR) algorithm.

        Returns
        ----------
        int
            1 if a proposed sample is accepted and 0 otherwise.
        """
        
        self.proposals[0]._adapt(self.x.point.squeeze())
        return super()._sample()
    

class MetropolisHastings(DelayedRejection):

    """
    The basic Metropolis-Hastings MCMC sampler. Samples are proposed with a Gaussian random walk and accepted with probability given by the Metropolis-Hastings acceptance kernel. This is a child of the `DelayedRejection` class.

    Attributes
    ----------
    The attributes are the same as those of the `DelayedRejection` parent class.
    """

    def __init__(
            self,
            target : Callable[[torch.Tensor], torch.Tensor],
            x0 : Union[Sample, torch.Tensor],
            sqrt_cov : torch.Tensor = None,
            cov : torch.Tensor = None,
            noisy : bool = False,
            reeval : int = 200
    ):
        
        """
        MetropolisHastings constructor

        target : Callable[[torch.Tensor], [torch.Tensor]]
            function evaluating the log probability of the target distribution
        sqrt_cov : torch.Tensor
            the lower cholesky decomposition of the covariance of the Gaussian random walk proposal
        cov : torch.Tensor
            the covariance of the Gaussian random walk proposal. Do not need to pass this if `sqrt_cov` is already passed.
        """
        
        super().__init__(
            target = target,
            x0 = x0,
            proposals = [GaussianRandomWalk(sqrt_cov, cov)],
            noisy = noisy,
            reeval = reeval
        )
