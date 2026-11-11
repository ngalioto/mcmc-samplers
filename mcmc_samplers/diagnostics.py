"""Convergence diagnostics for MCMC chains.

Pure-PyTorch implementations (no numpy/scipy/arviz dependency) of
autocorrelation, effective sample size, and R-hat.

Reference: Vehtari, Gelman, Simpson, Carpenter, Buerkner (2021).
"Rank-normalization, folding, and localization: an improved R-hat for
assessing convergence of MCMC." Bayesian Analysis 16(2): 667-718.
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


def _as_chains_array(chains: torch.Tensor) -> torch.Tensor:
    """Coerce ``(N,)``, ``(N, dim)`` or ``(num_chains, N, dim)`` to 3-D."""
    if chains.ndim == 3:
        return chains
    return _as_chain_array(chains).unsqueeze(0)


def _ess(samples: torch.Tensor) -> torch.Tensor:
    """Multi-chain ESS of a float64 ``(M, N, dim)`` array (Vehtari et al. 2021)."""
    M, N, dim = samples.shape
    chain_vars = samples.var(dim=1, unbiased=True)  # (M, dim)
    W = chain_vars.mean(dim=0)
    if M > 1:
        B = N * samples.mean(dim=1).var(dim=0, unbiased=True)
    else:
        B = torch.zeros(dim, dtype=samples.dtype)
    var_plus = (N - 1) / N * W + B / N

    # Biased autocovariance of each chain, averaged over chains: (N, dim)
    acov = torch.stack(
        [
            autocorrelation(samples[m], max_lag=N - 1) * chain_vars[m] * (N - 1) / N
            for m in range(M)
        ]
    ).mean(dim=0)

    ess = torch.zeros(dim, dtype=samples.dtype)
    for d in range(dim):
        if var_plus[d] <= 0:
            ess[d] = float(M * N)
            continue
        rho = 1.0 - (W[d] - acov[:, d]) / var_plus[d]
        rho[0] = 1.0
        # Geyer's initial monotone sequence over pairs of lags.
        rho_pair = rho[: 2 * (N // 2)].reshape(-1, 2).sum(dim=1)
        positive = torch.cumprod((rho_pair > 0).to(torch.int64), dim=0).bool()
        rho_pair = rho_pair[positive]
        for k in range(1, rho_pair.shape[0]):
            rho_pair[k] = min(rho_pair[k], rho_pair[k - 1])
        tau = max(-1.0 + 2.0 * rho_pair.sum().item(), 1.0)
        ess[d] = M * N / tau
    return ess


def effective_sample_size(chain: torch.Tensor) -> torch.Tensor:
    """Effective sample size using Geyer's initial monotone sequence estimator.

    Parameters
    ----------
    chain : torch.Tensor
        A single chain, ``(N,)`` or ``(N, dim)``, or several chains,
        ``(num_chains, N, dim)``. For several chains the combined multi-chain
        ESS of Vehtari et al. (2021) is returned, which is lowered when the
        chains disagree with each other.

    Returns
    -------
    torch.Tensor
        ESS per dimension, shape ``(dim,)`` (or scalar for univariate input).
    """
    ess = _ess(_as_chains_array(chain).to(torch.float64))
    if chain.ndim == 1:
        return ess.squeeze().to(chain.dtype)
    return ess.to(chain.dtype)


def _check_chains(chains: torch.Tensor) -> torch.Tensor:
    """Coerce ``(num_chains, N)`` or ``(num_chains, N, dim)`` to float64 3-D."""
    if chains.ndim == 2:
        chains = chains.unsqueeze(-1)
    if chains.ndim != 3:
        raise ValueError(
            f"chains must have shape (num_chains, N) or (num_chains, N, dim), "
            f"got {tuple(chains.shape)}"
        )
    return chains.to(torch.float64)


def _basic_rhat(samples: torch.Tensor) -> torch.Tensor:
    """Classic R-hat of a float64 ``(M, N, dim)`` array."""
    M, N, _ = samples.shape
    W = samples.var(dim=1, unbiased=True).mean(dim=0)
    B = N * samples.mean(dim=1).var(dim=0, unbiased=True)
    var_hat = (N - 1) / N * W + B / N
    safe_W = torch.where(W > 0, W, torch.ones_like(W))
    return torch.sqrt(var_hat / safe_W)


def _squeeze_univariate(out: torch.Tensor, chains: torch.Tensor) -> torch.Tensor:
    if chains.ndim == 2:
        return out.squeeze().to(chains.dtype)
    return out.to(chains.dtype)


def gelman_rubin(chains: torch.Tensor) -> torch.Tensor:
    """Classic Gelman-Rubin (1992) potential scale reduction factor (R-hat).

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
    Prefer :func:`rhat`, the rank-normalized split R-hat, which also detects
    drift within a chain and chains that differ in scale or tails.
    """
    samples = _check_chains(chains)
    if samples.shape[0] < 2:
        raise ValueError("Gelman-Rubin requires at least 2 chains.")
    return _squeeze_univariate(_basic_rhat(samples), chains)


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks of each column of ``x``, averaging over ties.

    Ties are common in MCMC output, since a rejected step repeats a draw.
    """
    ranks = torch.empty_like(x)
    for d in range(x.shape[1]):
        values, order = torch.sort(x[:, d])
        _, counts = torch.unique_consecutive(values, return_counts=True)
        ends = torch.cumsum(counts, dim=0).to(x.dtype)
        avg = ends - (counts.to(x.dtype) - 1) / 2
        ranks[order, d] = torch.repeat_interleave(avg, counts)
    return ranks


def _rank_normalize(samples: torch.Tensor) -> torch.Tensor:
    """Replace draws by the normal scores of their ranks across all chains."""
    M, N, dim = samples.shape
    S = M * N
    ranks = _average_ranks(samples.reshape(S, dim))
    return torch.special.ndtri((ranks - 0.375) / (S + 0.25)).reshape(M, N, dim)


def rhat(chains: torch.Tensor) -> torch.Tensor:
    """Rank-normalized split R-hat (Vehtari et al. 2021).

    Each chain is split in half (to detect drift within a chain), draws are
    replaced by normal scores of their ranks (robust to heavy tails), and the
    larger of the bulk R-hat and the tail R-hat (computed on ``|x - median|``)
    is returned. Values above about 1.01 suggest the chains have not mixed.

    Parameters
    ----------
    chains : torch.Tensor
        Tensor of shape ``(num_chains, N, dim)`` or ``(num_chains, N)``.
        A single chain is allowed, since it is split in two.

    Returns
    -------
    torch.Tensor
        R-hat per dimension, shape ``(dim,)`` or scalar for univariate input.
    """
    samples = _check_chains(chains)
    M, N, dim = samples.shape
    half = N // 2
    if half < 2:
        raise ValueError("rhat needs at least 4 draws per chain.")
    split = torch.cat([samples[:, :half], samples[:, N - half :]], dim=0)

    bulk = _basic_rhat(_rank_normalize(split))
    median = split.reshape(-1, dim).median(dim=0).values
    tail = _basic_rhat(_rank_normalize((split - median).abs()))
    return _squeeze_univariate(torch.maximum(bulk, tail), chains)


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
        Keys: ``mean``, ``std``, ``ess`` and, for multi-chain input,
        ``rhat`` (rank-normalized split R-hat). All values are 1-D tensors of
        length ``dim``.
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
    return {
        "mean": flat.mean(dim=0).to(chains.dtype),
        "std": flat.std(dim=0, unbiased=True).to(chains.dtype),
        "ess": effective_sample_size(multi).to(chains.dtype),
        "rhat": rhat(multi).to(chains.dtype),
    }
