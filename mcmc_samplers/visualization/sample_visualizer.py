import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

"""
Plotting functions for visualizing sample distributions.
"""

class SampleVisualizer:

    """
    Class to hold and manipulate samples for plotting.

    Attributes
    ----------
    samples : torch.Tensor
        Samples from target distribution.
    dim : int
        Dimension of samples.
    """

    def __init__(
            self,
            samples : torch.Tensor
    ):

        """
        Sample_Visualizer constructor.

        Parameters
        ----------
        samples : torch.Tensor
            2D tensor holding samples from target distribution. First dimension should be number of samples and second dimension should be sample dimension.
        """

        self.samples = samples.detach()
        self.dim = samples.shape[1]

    def _get_defaults(
            self,
            labels : Tuple[str, ...],
            titles : Tuple[str, ...],
            partition : Tuple[Tuple[int, ...], ...]
    ):
        if partition is None:
            partition = [range(self.dim)]
        if labels is None:
            labels = [f'$\\theta_{ii}$' for ii in range(self.dim)]
        if titles is None:
            titles = [None for ii in range(len(partition))]
        else:
            assert len(titles) == len(partition)
        return labels, titles, partition
        

    def chains(
            self,
            step : int = 1,
            labels : Tuple[str, ...] = None,
            titles : Tuple[str, ...] = None,
            partition : Tuple[Tuple[int, ...], ...] = None,
    ) -> Tuple[Tuple[plt.Figure, ...], Tuple[np.ndarray, ...]]:

        """
        Plots the sample chains.

        Parameters
        ----------
        step : int
            Step size by which to subsample the Markov chain. Default is 1, which results in no subsampling.
        labels : Tuple[str, ...]
            Iterable containing a label for each sample component. The length should be equal to the sample dimension. By default, the $i$th sample is labeled as $\theta_i$.
        titles : Tuple[str, ...]
           Iterable containing a suptitle for each figure. By default, empty strings are used for figure titles.
        partition : Tuple[Tuple[int, ...], ...]
            Iterable containing iterables of indices. Each iterable produces a plot containing the samples that correspond to the indices within that iterable. The indices refer to the sample dimension and not the chain iteration. By default, all samples are plotted in a single plot by setting `partition=[range(self.dim)]'

        Returns
        ----------
        Tuple[Tuple[plt.Figure, ...], Tuple[np.ndarray, ...]]
            The first Tuple is a list of Figure objects corresponding to each plot. The second Tuple is a list numpy arrays of the Axes objects from each plot. The length of each list equals the length of `partition`.
        """

        labels, titles, partition = self._get_defaults(labels,titles,partition)

        figs = []
        axes = []
        xx = range(0, self.samples.shape[0], step)

        for ii, idx in enumerate(partition):
            fig, axs = plt.subplots(len(idx), sharex=True)
            for jj, idx_jj in enumerate(idx):
                axs[jj].plot(xx, self.samples[::step, idx_jj])
                axs[jj].set_ylabel(labels[idx_jj])

            axs[-1].set_xlabel('Iteration')
            fig.suptitle(titles[ii])
            figs.append(fig)
            axes.append(axs)

        return figs, axes



    def triangular_hist(
            self,
            bins : int = None,
            step : int = 1,
            labels : Tuple[str, ...] = None,
            titles : Tuple[str, ...] = None,
            partition : Tuple[Tuple[int, ...], ...] = None,
    ) -> Tuple[Tuple[plt.Figure, ...], Tuple[np.ndarray, ...]]:
        
        """
        Plots 1D and 2D histograms of the samples.

        Parameters
        ----------
        bins : int
            Number of bins to use along each axis in each histogram. Default is 10.
        step : int
            Step size by which to subsample the Markov chain. Default is 1, which results in no subsampling.
        labels : Tuple[str, ...]
            Iterable containing a label for each sample component. The length should be equal to the sample dimension. By default, the $i$th sample is labeled as $\theta_i$.
        titles : Tuple[str, ...]
           Iterable containing a suptitle for each figure. By default, empty strings are used for figure titles.
        partition : Tuple[Tuple[int, ...], ...]
            Iterable containing iterables of indices. Each iterable produces a plot containing the samples that correspond to the indices within that iterable. The indices refer to the sample dimension and not the chain iteration. By default, all samples are plotted in a single plot by setting `partition=[range(self.dim)]'

        Returns
        ----------
        Tuple[Tuple[plt.Figure, ...], Tuple[np.ndarray, ...]]
            The first Tuple is a list of Figure objects corresponding to each plot. The second Tuple is a list numpy arrays of the Axes objects from each plot. The length of each list equals the length of `partition`.
        """

        labels, titles, partition = self._get_defaults(labels,titles,partition)

        figs = []
        axes = []

        for ii, idx in enumerate(partition):
            len_idx = len(idx)
            fig, axs = plt.subplots(len_idx, len_idx, squeeze=False)
            for row in range(len_idx):
                for col in range(len_idx):
                    # upper triangle. no plots.
                    if row < col:
                        axs[row,col].axis('off')
                    # diagonal. 1D histograms.
                    elif row == col:
                        axs[row,col].hist(self.samples[::step,idx[row]], bins=bins)
                        plt.setp(axs[row,col].get_yaxis(), visible=False)        
                    # lower triangle. 2D histograms.
                    else:
                        # share x-axis with 1D histogram from same column.
                        axs[row-1,col].get_shared_x_axes().join(axs[row-1,col],axs[row,col])
                        # share y-axis with 2D histograms from same row.
                        if col > 0:
                            axs[row,col-1].get_shared_y_axes().join(axs[row,col-1],axs[row,col])
                        
                        axs[row,col].hist2d(self.samples[::step,idx[col]], self.samples[::step,idx[row]], bins=bins)

                    # Adjust axis labels and axis visibility.
                    if col == 0 and row > 0:
                        axs[row,col].set_ylabel(labels[idx[row]])
                    else:
                        plt.setp(axs[row,col].get_yticklabels(), visible=False)        
                    if row == len(idx)-1:
                        axs[row,col].set_xlabel(labels[idx[col]])
                    else:
                        plt.setp(axs[row,col].get_xticklabels(), visible=False)

            fig.suptitle(titles[ii])
            figs.append(fig)
            axes.append(axs)
            
        return figs, axes
