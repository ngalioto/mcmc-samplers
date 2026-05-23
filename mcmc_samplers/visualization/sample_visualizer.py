"""Plotting utilities for MCMC samples."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..diagnostics import autocorrelation, effective_sample_size, gelman_rubin


class SampleVisualizer:
    """Plot and summarize an MCMC chain (single- or multi-chain).

    Parameters
    ----------
    samples : torch.Tensor
        Either ``(N, dim)`` for a single chain or ``(num_chains, N, dim)`` for
        multiple chains. Univariate ``(N,)`` is also accepted.
    """

    def __init__(self, samples: torch.Tensor):
        samples = samples.detach()
        if samples.ndim == 1:
            samples = samples.unsqueeze(-1)
            self._multi = False
        elif samples.ndim == 2:
            self._multi = False
        elif samples.ndim == 3:
            self._multi = True
        else:
            raise ValueError(
                f"samples must have 1, 2, or 3 dims, got {samples.ndim}"
            )
        self.samples = samples
        self.dim = int(samples.shape[-1])

    @property
    def _flat(self) -> torch.Tensor:
        """Single-chain view (samples stacked across chains)."""
        if self._multi:
            return self.samples.reshape(-1, self.dim)
        return self.samples

    def _get_defaults(
        self,
        labels: Tuple[str, ...] | None,
        titles: Tuple[str, ...] | None,
        partition: Tuple[Tuple[int, ...], ...] | None,
    ):
        if partition is None:
            partition = [range(self.dim)]
        if labels is None:
            labels = [f"$\\theta_{{{ii}}}$" for ii in range(self.dim)]
        if titles is None:
            titles = [None for _ in range(len(partition))]
        else:
            assert len(titles) == len(partition)
        return labels, titles, partition

    def chains(
        self,
        step: int = 1,
        labels: Tuple[str, ...] | None = None,
        titles: Tuple[str, ...] | None = None,
        partition: Tuple[Tuple[int, ...], ...] | None = None,
    ):
        """Plot the trace of each chain (per dimension)."""
        labels, titles, partition = self._get_defaults(labels, titles, partition)

        figs, axes_out = [], []
        if self._multi:
            num_chains, N, _ = self.samples.shape
            samples_per_chain = self.samples
        else:
            N = self.samples.shape[0]
            samples_per_chain = self.samples.unsqueeze(0)
            num_chains = 1

        xx = range(0, N, step)
        for ii, idx in enumerate(partition):
            fig, axs = plt.subplots(len(idx), sharex=True, squeeze=False)
            axs = axs[:, 0]
            for jj, idx_jj in enumerate(idx):
                for c in range(num_chains):
                    axs[jj].plot(
                        xx,
                        samples_per_chain[c, ::step, idx_jj],
                        alpha=0.6 if num_chains > 1 else 1.0,
                    )
                axs[jj].set_ylabel(labels[idx_jj])
            axs[-1].set_xlabel("Iteration")
            fig.suptitle(titles[ii])
            figs.append(fig)
            axes_out.append(axs)

        return figs, axes_out

    def triangular_hist(
        self,
        bins: int | None = None,
        step: int = 1,
        labels: Tuple[str, ...] | None = None,
        titles: Tuple[str, ...] | None = None,
        partition: Tuple[Tuple[int, ...], ...] | None = None,
        **hist_kwargs,
    ):
        """Plot a triangular matrix of 1-D / 2-D marginal histograms."""
        labels, titles, partition = self._get_defaults(labels, titles, partition)

        samples = self._flat
        figs, axes_out = [], []
        for ii, idx in enumerate(partition):
            len_idx = len(idx)
            fig, axs = plt.subplots(len_idx, len_idx, squeeze=False)
            for row in range(len_idx):
                for col in range(len_idx):
                    if row < col:
                        axs[row, col].axis("off")
                    elif row == col:
                        axs[row, col].hist(
                            samples[::step, idx[row]], bins=bins, **hist_kwargs
                        )
                        plt.setp(axs[row, col].get_yaxis(), visible=False)
                    else:
                        axs[row - 1, col].sharex(axs[row, col])
                        if col > 0:
                            axs[row, col - 1].sharey(axs[row, col])
                        axs[row, col].hist2d(
                            samples[::step, idx[col]].numpy(),
                            samples[::step, idx[row]].numpy(),
                            bins=bins,
                            **hist_kwargs,
                        )

                    if col == 0 and row > 0:
                        axs[row, col].set_ylabel(labels[idx[row]])
                    else:
                        plt.setp(axs[row, col].get_yticklabels(), visible=False)
                    if row == len(idx) - 1:
                        axs[row, col].set_xlabel(labels[idx[col]])
                    else:
                        plt.setp(axs[row, col].get_xticklabels(), visible=False)

            fig.suptitle(titles[ii])
            figs.append(fig)
            axes_out.append(axs)

        return figs, axes_out

    def autocorrelation(
        self,
        max_lag: int | None = None,
        labels: Tuple[str, ...] | None = None,
        titles: Tuple[str, ...] | None = None,
        partition: Tuple[Tuple[int, ...], ...] | None = None,
    ):
        """Plot per-dimension autocorrelation functions.

        For multi-chain input, all chains are overlaid on the same axes.
        """
        labels, titles, partition = self._get_defaults(labels, titles, partition)

        if self._multi:
            acfs = torch.stack(
                [autocorrelation(self.samples[c], max_lag=max_lag) for c in range(self.samples.shape[0])]
            )
        else:
            acfs = autocorrelation(self.samples, max_lag=max_lag).unsqueeze(0)
        lags = range(acfs.shape[1])

        figs, axes_out = [], []
        for ii, idx in enumerate(partition):
            fig, axs = plt.subplots(len(idx), sharex=True, squeeze=False)
            axs = axs[:, 0]
            for jj, idx_jj in enumerate(idx):
                for c in range(acfs.shape[0]):
                    axs[jj].plot(
                        lags,
                        acfs[c, :, idx_jj],
                        alpha=0.6 if acfs.shape[0] > 1 else 1.0,
                    )
                axs[jj].axhline(0.0, color="k", linewidth=0.5)
                axs[jj].set_ylabel(labels[idx_jj])
            axs[-1].set_xlabel("Lag")
            fig.suptitle(titles[ii])
            figs.append(fig)
            axes_out.append(axs)

        return figs, axes_out

    def summary(self, labels: Tuple[str, ...] | None = None) -> str:
        """Return a printable summary of mean, sd, ESS, and R-hat (multi-chain)."""
        if labels is None:
            labels = [f"theta[{ii}]" for ii in range(self.dim)]

        flat = self._flat
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=True)

        if self._multi:
            ess_per_chain = torch.stack(
                [
                    effective_sample_size(self.samples[c])
                    for c in range(self.samples.shape[0])
                ]
            )
            ess = ess_per_chain.sum(dim=0)
            rhat = gelman_rubin(self.samples)
        else:
            ess = effective_sample_size(self.samples)
            rhat = None

        rows = [f"{'param':<12} {'mean':>10} {'sd':>10} {'ess':>10}"]
        if rhat is not None:
            rows[0] += f" {'rhat':>8}"
        for d in range(self.dim):
            row = (
                f"{labels[d]:<12} {mean[d].item():>10.4f} "
                f"{std[d].item():>10.4f} {ess[d].item():>10.1f}"
            )
            if rhat is not None:
                row += f" {rhat[d].item():>8.4f}"
            rows.append(row)
        return "\n".join(rows)
