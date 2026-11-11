"""Plotting utilities for MCMC samples."""

from __future__ import annotations

from typing import Tuple

import torch

try:  # matplotlib is an optional dependency: pip install mcmc-samplers[viz]
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .. import diagnostics
from ..diagnostics import autocorrelation, effective_sample_size


class SampleVisualizer:
    """Plot and summarize an MCMC chain (single- or multi-chain).

    Parameters
    ----------
    samples : torch.Tensor
        Either ``(N, dim)`` for a single chain or ``(num_chains, N, dim)`` for
        multiple chains. Univariate ``(N,)`` is also accepted.
    """

    def __init__(self, samples: torch.Tensor):
        if plt is None:
            raise ImportError(
                "SampleVisualizer requires matplotlib: pip install mcmc-samplers[viz]"
            )
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
        fig=None,
    ):
        """Plot the trace of each chain (per dimension).

        Pass ``fig`` (a Figure or SubFigure) to draw into it instead of a new
        figure, e.g. to compare methods side by side with ``fig.subfigures``.
        """
        labels, titles, partition = self._get_defaults(labels, titles, partition)
        if fig is not None and len(partition) > 1:
            raise ValueError("fig can only be used with a single partition")

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
            if fig is None:
                fig = plt.figure()
            axs = fig.subplots(len(idx), sharex=True, squeeze=False)[:, 0]
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
            fig = None

        return figs, axes_out

    def triangular_hist(
        self,
        bins: int | None = None,
        step: int = 1,
        labels: Tuple[str, ...] | None = None,
        titles: Tuple[str, ...] | None = None,
        partition: Tuple[Tuple[int, ...], ...] | None = None,
        fig=None,
        **hist_kwargs,
    ):
        """Plot a triangular matrix of 1-D / 2-D marginal histograms.

        Pass ``fig`` (a Figure or SubFigure) to draw into it instead of a new
        figure, e.g. to compare methods side by side with ``fig.subfigures``.
        """
        labels, titles, partition = self._get_defaults(labels, titles, partition)
        if fig is not None and len(partition) > 1:
            raise ValueError("fig can only be used with a single partition")

        samples = self._flat
        figs, axes_out = [], []
        for ii, idx in enumerate(partition):
            len_idx = len(idx)
            if fig is None:
                fig = plt.figure()
            axs = fig.subplots(len_idx, len_idx, squeeze=False)
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
            fig = None

        return figs, axes_out

    def autocorrelation(
        self,
        max_lag: int | None = None,
        labels: Tuple[str, ...] | None = None,
        titles: Tuple[str, ...] | None = None,
        partition: Tuple[Tuple[int, ...], ...] | None = None,
        axs=None,
        label: str | None = None,
    ):
        """Plot per-dimension autocorrelation functions.

        For multi-chain input, all chains are overlaid on the same axes.
        To compare methods on one plot, pass the ``axs`` returned by a previous
        call and give each call a ``label`` for the legend.
        """
        labels, titles, partition = self._get_defaults(labels, titles, partition)
        if axs is not None and len(partition) > 1:
            raise ValueError("axs can only be used with a single partition")

        if self._multi:
            acfs = torch.stack(
                [autocorrelation(self.samples[c], max_lag=max_lag) for c in range(self.samples.shape[0])]
            )
        else:
            acfs = autocorrelation(self.samples, max_lag=max_lag).unsqueeze(0)
        lags = range(acfs.shape[1])

        figs, axes_out = [], []
        for ii, idx in enumerate(partition):
            if axs is None:
                fig, axs = plt.subplots(len(idx), sharex=True, squeeze=False)
                axs = axs[:, 0]
            fig = axs[0].figure
            for jj, idx_jj in enumerate(idx):
                color = None  # one colour per call, shared by all its chains
                for c in range(acfs.shape[0]):
                    (line,) = axs[jj].plot(
                        lags,
                        acfs[c, :, idx_jj],
                        color=color,
                        alpha=0.6 if acfs.shape[0] > 1 else 1.0,
                        label=label if c == 0 else None,
                    )
                    color = line.get_color()
                axs[jj].axhline(0.0, color="k", linewidth=0.5)
                axs[jj].set_ylabel(labels[idx_jj])
                if label is not None:
                    axs[jj].legend()
            axs[-1].set_xlabel("Lag")
            if titles[ii] is not None:
                fig.suptitle(titles[ii])
            figs.append(fig)
            axes_out.append(axs)
            axs = None

        return figs, axes_out

    def summary(self, labels: Tuple[str, ...] | None = None) -> str:
        """Return a printable summary of mean, sd, ESS, and R-hat (multi-chain)."""
        if labels is None:
            labels = [f"theta[{ii}]" for ii in range(self.dim)]

        flat = self._flat
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=True)

        ess = effective_sample_size(self.samples)
        rhat = diagnostics.rhat(self.samples) if self._multi else None

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
