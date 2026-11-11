"""Tests for the diagnostics module."""

from __future__ import annotations

import torch

from mcmc_samplers import diagnostics


def test_autocorrelation_iid_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(5000, 2)
    acf = diagnostics.autocorrelation(x, max_lag=20)
    assert acf.shape == (21, 2)
    # Lag 0 should be 1 by construction.
    assert torch.allclose(acf[0], torch.ones(2), atol=1e-6)
    # Higher lags should be near zero for iid data.
    assert acf[1:].abs().max() < 0.15


def test_autocorrelation_ar1_decays_geometrically():
    torch.manual_seed(0)
    rho = 0.6
    N = 10_000
    eps = torch.randn(N)
    x = torch.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]
    acf = diagnostics.autocorrelation(x, max_lag=10)
    assert acf.shape == (11,)
    expected = torch.tensor([rho ** k for k in range(11)])
    assert torch.allclose(acf, expected, atol=0.1)


def test_ess_for_iid_close_to_N():
    torch.manual_seed(0)
    x = torch.randn(5000)
    ess = diagnostics.effective_sample_size(x)
    assert 4000 < float(ess.item()) <= 5000


def test_ess_drops_for_high_autocorrelation():
    torch.manual_seed(0)
    rho = 0.9
    N = 5000
    eps = torch.randn(N)
    x = torch.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]
    ess = diagnostics.effective_sample_size(x)
    # For AR(1) with rho = 0.9, integrated autocorrelation tau ~ (1+rho)/(1-rho) = 19
    # so ESS ~ N / 19 ~ 263.
    assert float(ess.item()) < N / 5


def test_gelman_rubin_one_for_identical_chains():
    torch.manual_seed(0)
    N = 500
    chain = torch.randn(N, 2)
    chains = chain.unsqueeze(0).repeat(3, 1, 1)
    rhat = diagnostics.gelman_rubin(chains)
    # Identical chains give B = 0, so R-hat = sqrt((N - 1) / N), just below 1.
    expected = torch.full((2,), ((N - 1) / N) ** 0.5)
    assert torch.allclose(rhat, expected, atol=1e-6)


def test_gelman_rubin_high_for_disjoint_chains():
    torch.manual_seed(0)
    chains = torch.stack(
        [
            torch.randn(500, 2) + 10.0,
            torch.randn(500, 2) - 10.0,
            torch.randn(500, 2),
        ]
    )
    rhat = diagnostics.gelman_rubin(chains)
    assert (rhat > 1.5).all()


def test_multichain_ess_matches_ar1_theory():
    torch.manual_seed(0)
    M, N, rho = 4, 4000, 0.8
    eps = torch.randn(M, N, 1)
    x = torch.zeros(M, N, 1)
    for t in range(1, N):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]
    expected = M * N * (1 - rho) / (1 + rho)
    ess = float(diagnostics.effective_sample_size(x))
    assert 0.8 * expected < ess < 1.2 * expected


def test_rhat_near_one_for_iid_chains():
    torch.manual_seed(0)
    assert abs(float(diagnostics.rhat(torch.randn(4, 1000))) - 1.0) < 0.01


def test_rhat_flags_what_classic_rhat_misses():
    torch.manual_seed(0)
    scale = torch.randn(4, 1000)
    scale[0] *= 3  # same mean, different spread
    drift = torch.randn(4, 1000) + torch.linspace(0, 2, 1000)  # drift within chains
    for chains in (scale, drift):
        assert float(diagnostics.gelman_rubin(chains)) < 1.01
        assert float(diagnostics.rhat(chains)) > 1.05


def test_rhat_handles_repeated_draws():
    torch.manual_seed(0)
    x = torch.randn(4, 1000)
    x[:, 1::2] = x[:, 0::2]  # every draw repeated, like MH rejections
    assert abs(float(diagnostics.rhat(x)) - 1.0) < 0.02
