import torch
from math import sqrt
from mcmc_samplers import Proposal, Sample
from typing import Tuple, Union

"""
Defines the `GaussianRandomWalk` base class and all proposals that are derived from it.
"""

class GaussianRandomWalk(Proposal):
    
    """
    Generates proposals using a Gaussian random walk.

    Attributes
    ---------
    sqrt_cov : torch.Tensor
        Lower cholesky decomposition of the covariance of Gaussian proposal
    dim : int
        Dimension of the Gaussian proposal
    """

    def __init__(
            self,
            sqrt_cov : torch.Tensor = None,
            cov : torch.Tensor = None
    ):
        
        """
        GaussianRandomWalk constructor.

        Parameters
        ----------
        sqrt_cov : torch.Tensor
            Lower cholesky decomposition of the covariance of Gaussian proposal. Must be 2-dimensional and square.
        cov : torch.Tensor
            Covariance matrix of the Gaussian proposal. Must be symmetric and positive-definite.
        """
        
        if sqrt_cov is not None:
            if sqrt_cov.ndim !=2 or sqrt_cov.shape[0] != sqrt_cov.shape[1]:
                raise ValueError(f'sqrt_cov must be a square matrix but instead it has shape {sqrt_cov.shape}')
            self.sqrt_cov = sqrt_cov
        elif cov is not None:
            try:
                self.sqrt_cov = torch.linalg.cholesky(cov, upper=False)
            except:
                raise ValueError('cov must be a symmetric positive-definite matrix')
        else:
            raise ValueError('GaussianRandomWalk requires either sqrt_cov or cov for initialization')

        self.dim = self.sqrt_cov.shape[0]
        
    @property
    def is_symmetric(
            self
    ) -> bool:
        
        """
        Getter of `is_symmetric` property. 

        Returns
        ----------
        bool
            Gaussian distributions are symmetric, so this always returns `True`
        """
        
        return True


    def propose(
            self,
            y : Tuple[Sample, ...],
            N : int = 1
    ) -> Union[Sample, Tuple[Sample, ...]]:
        
        """
        Draws a sample from the Gaussian random walk

        Parameters
        ----------
        y : Tuple[Sample, ...]
            The Gaussian distribution centers itself around the first Sample in this list
        N : int
            Number of samples to draw. Default is 1

        Returns
        ----------
        Union[Sample, Tuple[Sample, ...]]
            The samples drawn from the Gaussian random walk.
            If `N` is 1, a Sample object is returned
            If `N`>1, a list of Sample objects is returned
        """
        
        sample_points = torch.randn(N, self.dim) @ self.sqrt_cov.T + y[0].point
        return [Sample(sample_points[ii]) for ii in range(N)] if N > 1 else Sample(sample_points)

    def log_prob(
            self,
            y : Tuple[Sample, ...]
    ) -> torch.Tensor:
        
        """
        Evaluates the log probability density of the Gaussian distribution

        Parameters
        ----------
        y : Tuple[Sample, ...]
            List of Samples representing the current state and all proposals for a given iteration ordered from first proposal tier to current proposal tier

        Returns
        ----------
        torch.Tensor
            Log probability density for a Gaussian centered at the first Sample in `y` evaluated at the last Sample in `y`.
        """
        
        return -0.5 * torch.sum(torch.linalg.solve_triangular(self.sqrt_cov, (y[0].point - y[-1].point).T, upper=False)**2, dim=0)


class AdaptiveCovariance(GaussianRandomWalk):

    """
    A Gaussian random walk with covariance that adapts according to the sample covariance. This is a child class of the `GaussianRandomWalk` class.

    Attributes
    ---------
    sqrt_cov : torch.Tensor
        Lower cholesky decomposition of the covariance of Gaussian proposal
    cov : torch.Tensor
        Covariance matrix of the Gaussian proposal
    dim : int
        Dimension of the Gaussian proposal
    sd : float
        Scaling factor of the proposal covariance.
    n0 : int
        Number of Markov chain iterations before using adaptive covariance
    eps : float
        Small positive value to add to covariance diagonal during adaptation to encourage positive-definiteness
    iteration : int
        The iteration number.
    mean : torch.Tensor
        The running sample mean
    """

    def __init__(
            self,
            cov : torch.Tensor,
            sd : float = None,
            n0 : int = 200,
            eps : float = 1e-8
    ):
        
        """
        AdaptiveCovariance constructor.

        Parameters
        ----------
        cov : torch.Tensor
            Covariance matrix of the Gaussian proposal. Must be 2-dimensional and square.
        sd : float
            Scaling factor of proposal covariance. If `None`, the default value is $2.4^2 / d$, where $d$ is the proposal dimension.
        n0 : int
            Number of Markov chain iterations before using adaptive covariance. Default is 200
        eps : float
            Small positive value to be added to the covariance diagonal during adaptation to encourage positive-definiteness. Default is 1e-8
        """

        super().__init__(cov = cov)
        self.sd = 2.4**2 / self.dim if sd is None else sd
        self.sqrt_cov *= sqrt(self.sd)
        self.cov = cov * self.sd
        self.n0 = n0
        self.eps = eps
        
        self.iteration = 0
        self.mean = torch.zeros(self.dim)

    @property
    def cov(
            self
    ) -> torch.Tensor:
        
        """
        Getter for the `cov` property.

        Returns
        ----------
        torch.Tensor
            If `iteration`>`n0`, the sample covariance is returned. Otherwise, the initial covariance is returned.
        """
        
        if self.iteration > self.n0:
            return self._cov_ii
        else:
            return self._cov_0

    @cov.setter
    def cov(
        self,
        value : torch.Tensor
    ):
        
        """
        Setter for the `cov` property

        Parameters
        ----------
        value : torch.Tensor
            A covariance matrix.
        """
        
        self._cov_0 = value
        self._cov_ii = value.clone()


    def _adapt(
            self,
            x : torch.Tensor
    ):
        
        """
        Adaptation algorithm for updating the adaptive covariance. Updates the sample mean `mean` and covariance `_cov_ii`

        Parameters
        ----------
        x : torch.Tensor
            The most recent state value in the Markov chain.
        """
        
        self.iteration += 1

        self._cov_ii *= ((self.iteration - 1) / self.iteration)
        outer = self.iteration * torch.outer(self.mean, self.mean) + torch.outer(x, x)
        
        self.mean = (self.mean * (self.iteration - 1) + x) / self.iteration
        
        outer_mean = (self.iteration + 1) * torch.outer(self.mean, self.mean)

        self._cov_ii += (self.sd / self.iteration) * (outer - outer_mean + self.eps * torch.eye(self.dim))

        if self.iteration > self.n0:
            try:
                self.sqrt_cov = torch.linalg.cholesky(self._cov_ii, upper=False)
            except:
                for ii in range(-8, 1, -1):
                    try:
                        self._cov_ii += torch.diag(10^(ii)*torch.ones(self.dim))
                        self.sqrt_cov = torch.linalg.cholesky(self._cov_ii, upper=False)
                        break
                    except:
                        pass

    def propose(
            self,
            y : Tuple[Sample, ...],
            N : int = 1
    ) -> Union[Sample, Tuple[Sample, ...]]:
        
        """
        Generates a proposed state from the Gaussian random walk with adaptive covariance.

        Parameters
        ----------
        y : Tuple[Sample, ...]
            A list of Sample objects containing the current state and proposals from each proposal tier
        N : int
            Number of proposed states to draw. Default is 1

        Returns
        ----------
        Union[Sample, Tuple[Sample, ...]]
            Returns the proposed state wrapped inside a Sample object if `N` is 1. Otherwise, returns a list of Sample objects, each containing a proposed state
        """
        
        return super().propose(y,N)


class ScaledCovariance(GaussianRandomWalk):

    """
    Wraps a GaussianRandomWalk
    """

    def __init__(
            self,
            covariance : GaussianRandomWalk,
            gamma : float = 1e-2
    ):
        
        """
        ScaledCovariance constructor.

        Parameters
        ----------
        covariance : GaussianRandomWalk
            The Gaussian random walk proposal to be scaled.
        gamma : float
            The scaling factor for the second-stage proposal covariance. Default is 1e-2.
        """
        
        self.covariance = covariance
        self.sqrt_gamma = sqrt(gamma)

    @property
    def sqrt_cov(
            self
    ) -> torch.Tensor:
        
        """
        Getter for the `sqrt_cov` property. 

        Returns
        ----------
        torch.Tensor
            The `sqrt_cov` property of the wrapped GaussianRandomWalk object scaled by `sqrt_gamma`.
        """
        
        return self.covariance.sqrt_cov * self.sqrt_gamma

    @property
    def dim(
            self
    ) -> int:
        
        """
        Getter for the `dim` property. 

        Returns
        ----------
        int
            The `dim` property of the wrapped GaussianRandomWalk object.
        """
        
        return self.covariance.dim
