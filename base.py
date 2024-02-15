import torch

'''
Base class used throughout the MCMC package
'''

class Sample:

    '''
    Data structure representing an MCMC sample.

    Attributes
    ----------
    point : torch.Tensor
        The sample value
    log_prob : torch.Tensor
        The log probability density of a distribution of interest evaluated at the sample value `point`. The distribution of interest is typically the target distribution of the MCMC algorithm.
    '''

    def __init__(
            self,
            point : torch.Tensor,
            log_prob : torch.Tensor = None
    ):
        '''
        Sample constructor

        Parameters
        ----------
        point : torch.Tensor
            sample value
        log_prob : torch.Tensor, optional
            log probability density of a distribution of interest evaluated at the sample value `point`.
        '''
        self.point = point
        self.log_prob = log_prob
