"""Convergence diagnostics for MCMC chains.

Pure-PyTorch implementations (no numpy/scipy/arviz dependency) of
autocorrelation, effective sample size, and Gelman-Rubin R-hat.
"""

from __future__ import annotations

import math

import torch


def _as_chain_array(chain: torch.Tensor) -> torch.Tensor:
    """Coerce a 1-D vector into a ``(N, 1)`` matrix; keep 2-D as is."""
    if chain.ndim == 1:
        return chain.unsqueeze(-1)
    if chain.ndim == 2:
        return chain
    raise ValueError(
        f"chain must be 1-D (univariate) or 2-D (multivariate), got shape "
        f"{tuple(chain.shape)}"
    )


def autocorrelation(
    chain: torch.Tensor,
    max_lag: int | None = None,
) -> torch.Tensor:
    """Per-dimension biased autocorrelation function, computed via FFT.

    Parameters
    ----------
    chain : torch.Tensor
        Either ``(N,)`` for a univariate chain or ``(N, dim)`` for multivariate.
    max_lag : int, optional
        Maximum lag to return. Defaults to ``N // 2``.

    Returns
    -------
    torch.Tensor
        Autocorrelation function of shape ``(max_lag + 1, dim)``, or
        ``(max_lag + 1,)`` for univariate ``(N,)`` input.
    """
    samples = _as_chain_array(chain).to(torch.float64)
    N, dim = samples.shape
    if max_lag is None:
        max_lag = N // 2

    centered = samples - samples.mean(dim=0, keepdim=True)
    n_fft = 1
    while n_fft < 2 * N:
        n_fft <<= 1
    padded = torch.zeros(n_fft, dim, dtype=samples.dtype)
    padded[:N] = centered
    fft_vals = torch.fft.fft(padded, dim=0)
    power = fft_vals * fft_vals.conj()
    acf_full = torch.fft.ifft(power, dim=0).real
    acf = acf_full[: max_lag + 1]
    var = acf[0]
    var = torch.where(var > 0, var, torch.ones_like(var))
    acf = acf / var
    if chain.ndim == 1:
        acf = acf.squeeze(-1)
    return acf.to(chain.dtype)


def effective_sample_size(chain: torch.Tensor) -> torch.Tensor:
    """Effective sample size using Geyer's initial monotone sequence estimator.

    Parameters
    ----------
    chain : torch.Tensor
        Univariate ``(N,)`` or multivariate ``(N, dim)`` chain.

    Returns
    -------
    torch.Tensor
        ESS per dimension, shape ``(dim,)`` (or scalar for univariate input).

    Notes
    -----
    Uses Geyer's (1992) initial monotone sequence estimator: sums sequential
    pairs of even/odd lag autocorrelations as long as they are positive and
    monotonically non-increasing. This is the standard estimator in
    PyMC/Stan/arviz for single-chain ESS.
    """
    samples = _as_chain_array(chain).to(torch.float64)
    N, dim = samples.shape
    acf = autocorrelation(samples, max_lag=N - 1)

    ess = torch.zeros(dim, dtype=torch.float64)
    for d in range(dim):
        rho = acf[:, d].clone()
        rho_pair = rho[: 2 * (rho.shape[0] // 2)].reshape(-1, 2).sum(dim=1)
        positive_mask = torch.cumprod((rho_pair > 0).to(torch.int64), dim=0).bool()
        if not positive_mask.any():
            ess[d] = float(N)
            continue
        rho_pair = rho_pair[positive_mask]
        for k in range(1, rho_pair.shape[0]):
            if rho_pair[k] > rho_pair[k - 1]:
                rho_pair[k] = rho_pair[k - 1]
        tau = -1.0 + 2.0 * rho_pair.sum().item()
        if tau < 1.0:
            tau = 1.0
        ess[d] = N / tau

    if chain.ndim == 1:
        return ess.squeeze().to(chain.dtype)
    return ess.to(chain.dtype)


def gelman_rubin(chains: torch.Tensor) -> torch.Tensor:
    """Gelman-Rubin (1992) potential scale reduction factor (R-hat).

    Parameters
    ----------
    chains : torch.Tensor
        Tensor of shape ``(num_chains, N, dim)`` or ``(num_chains, N)``.

    Returns
    -------
    torch.Tensor
        R-hat per dimension, shape ``(dim,)`` for multivariate input or
        scalar for univariate.

    Notes
    -----
    Returns the un-corrected (non rank-normalized) R-hat. Values close to 1.0
    indicate convergence; values above 1.1 are commonly flagged as cause for
    concern.
    """
    if chains.ndim == 2:
        chains = chains.unsqueeze(-1)
    if chains.ndim != 3:
        raise ValueError(
            f"chains must have shape (num_chains, N) or (num_chains, N, dim), "
            f"got {tuple(chains.shape)}"
        )

    samples = chains.to(torch.float64)
    M, N, dim = samples.shape
    if M < 2:
        raise ValueError("Gelman-Rubin requires at least 2 chains.")

    chain_means = samples.mean(dim=1)
    chain_vars = samples.var(dim=1, unbiased=True)

    grand_mean = chain_means.mean(dim=0)
    B = N / (M - 1) * ((chain_means - grand_mean) ** 2).sum(dim=0)
    W = chain_vars.mean(dim=0)

    safe_W = torch.where(W > 0, W, torch.ones_like(W))
    var_hat = (N - 1) / N * W + B / N
    rhat = torch.sqrt(var_hat / safe_W)

    if chains.shape[-1] == 1:
        return rhat.squeeze().to(chains.dtype)
    return rhat.to(chains.dtype)


def summarize(
    chain: torch.Tensor | None = None,
    chains: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a dictionary of standard chain summaries.

    Exactly one of ``chain`` (single chain ``(N, dim)``) or ``chains``
    (multi-chain ``(num_chains, N, dim)``) must be supplied.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``ess`` (per chain, averaged), and ``rhat``
        (only when multi-chain). All values are 1-D tensors of length ``dim``.
    """
    if (chain is None) == (chains is None):
        raise ValueError("Provide exactly one of `chain` or `chains`.")

    if chain is not None:
        arr = _as_chain_array(chain)
        return {
            "mean": arr.mean(dim=0),
            "std": arr.std(dim=0, unbiased=True),
            "ess": effective_sample_size(arr),
        }
    multi = chains.to(torch.float64)
    if multi.ndim == 2:
        multi = multi.unsqueeze(-1)
    flat = multi.reshape(-1, multi.shape[-1])
    ess_per_chain = torch.stack(
        [effective_sample_size(multi[i]) for i in range(multi.shape[0])],
        dim=0,
    )
    return {
        "mean": flat.mean(dim=0).to(chains.dtype),
        "std": flat.std(dim=0, unbiased=True).to(chains.dtype),
        "ess": ess_per_chain.sum(dim=0).to(chains.dtype),
        "rhat": gelman_rubin(multi),
    }
