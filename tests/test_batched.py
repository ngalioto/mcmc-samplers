"""Running many chains at once with ``batched=True`` (MH, AM, DRAM)."""

import pytest
import torch

from mcmc_samplers import (
    AdaptiveMetropolis,
    DelayedRejectionAdaptiveMetropolis,
    MetropolisHastings,
    diagnostics,
)

SAMPLERS = {
    "MH": lambda target, x0, gen, **kw: MetropolisHastings(target, x0, cov=torch.eye(2), generator=gen, **kw),
    "AM": lambda target, x0, gen, **kw: AdaptiveMetropolis(target, x0, cov=torch.eye(2), n0=50, generator=gen, **kw),
    "DRAM": lambda target, x0, gen, **kw: DelayedRejectionAdaptiveMetropolis(
        target, x0, cov=torch.eye(2), n0=50, generator=gen, **kw
    ),
}


@pytest.mark.parametrize("name", SAMPLERS)
def test_one_batched_chain_matches_unbatched_exactly(name, correlated_gaussian):
    """With one chain and the same seed, both code paths draw the same random
    numbers in the same order, so the chains must be identical."""
    make = SAMPLERS[name]
    single = make(correlated_gaussian, torch.zeros(2), torch.Generator().manual_seed(0))
    batch = make(correlated_gaussian, torch.zeros(1, 2), torch.Generator().manual_seed(0), batched=True)

    s1, lp1 = single(300, show_progress=False)
    s2, lp2 = batch(300, show_progress=False)
    assert s2.shape == (1, 300, 2) and lp2.shape == (1, 300)
    assert torch.equal(s1, s2[0]) and torch.equal(lp1, lp2[0])
    assert single.acceptance_ratio == pytest.approx(batch.acceptance_ratio)


def test_batched_dram_samples_target(correlated_gaussian):
    num_chains = 16
    sampler = DelayedRejectionAdaptiveMetropolis(
        correlated_gaussian,
        x0=3 * torch.randn(num_chains, 2, generator=torch.Generator().manual_seed(1)),
        cov=torch.eye(2),
        n0=100,
        generator=torch.Generator().manual_seed(0),
        batched=True,
    )
    chains, log_probs = sampler(2000, burn_in=500, show_progress=False)
    assert chains.shape == (num_chains, 1500, 2)
    assert log_probs.shape == (num_chains, 1500)

    flat = chains.reshape(-1, 2)
    assert torch.allclose(flat.mean(0), correlated_gaussian.mean, atol=0.1)
    assert torch.allclose(torch.cov(flat.T), correlated_gaussian.cov, atol=0.2)
    assert (diagnostics.rhat(chains) < 1.05).all()
    # Chains move independently: they should not all accept on the same steps.
    assert not torch.equal(chains[0], chains[1])


def test_batched_requires_2d_state(correlated_gaussian):
    with pytest.raises(ValueError):
        MetropolisHastings(correlated_gaussian, torch.zeros(2), cov=torch.eye(2), batched=True)


def test_batched_noisy_runs(correlated_gaussian):
    sampler = MetropolisHastings(
        correlated_gaussian, torch.zeros(4, 2), cov=100 * torch.eye(2),
        generator=torch.Generator().manual_seed(0), noisy=True, reeval=5, batched=True,
    )
    chains, log_probs = sampler(200, show_progress=False)
    assert torch.isfinite(chains).all() and torch.isfinite(log_probs).all()


def test_batched_resume_matches_uninterrupted_run(correlated_gaussian):
    def make():
        return DelayedRejectionAdaptiveMetropolis(
            correlated_gaussian, torch.zeros(3, 2), cov=torch.eye(2), n0=20,
            generator=torch.Generator().manual_seed(0), batched=True,
        )

    full = make()
    full(100, show_progress=False)
    expected, _ = full(100, show_progress=False)

    first = make()
    first(100, show_progress=False)
    resumed = make()
    resumed.load_state_dict(first.state_dict())
    got, _ = resumed(100, show_progress=False)
    assert torch.equal(expected, got)
