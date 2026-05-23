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
    chain = torch.randn(500, 2)
    chains = chain.unsqueeze(0).repeat(3, 1, 1)
    rhat = diagnostics.gelman_rubin(chains)
    assert torch.allclose(rhat, torch.ones(2), atol=1e-6)


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
