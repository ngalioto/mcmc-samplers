import torch

"""
Base class used throughout the 'mcmc-samplers' package
"""

class Sample:

    """
    Data structure representing an MCMC sample.

    Attributes
    ----------
    point : torch.Tensor
        The sample value
    log_prob : torch.Tensor
        The log probability density of a distribution of interest evaluated at the sample value `point`. The distribution of interest is typically the target distribution of the MCMC algorithm.
    """

    def __init__(
            self,
            point : torch.Tensor = None,
            log_prob : torch.Tensor = None
    ):
        
        """
        Sample constructor

        Parameters
        ----------
        point : torch.Tensor
            sample value
        log_prob : torch.Tensor, optional
            log probability density of a distribution of interest evaluated at the sample value `point`.
        """

        self.point = point
        self.log_prob = log_prob

    @property
    def shape(
            self
    ):
        return self.point.shape

    @shape.setter
    def shape(
            self,
            value
    ):
        raise AttributeError("attribute 'shape' of 'mcmc-samplers.Sample' objects is not writable")

    def __str__(
            self
    ) -> str:
        string = f'sample(point={self.point})'
        if self.log_prob is not None:
            string += f', log_prob={self.log_prob}'
        return string

    def __len__(
            self
    ) -> int:
        return len(self.point)
