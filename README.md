# MCMC Samplers

A Python package for performing MCMC sampling with PyTorch. Version 0.1.0.

## What's in this repo

- **Random-walk samplers**: Metropolis-Hastings, Adaptive Metropolis, and Delayed Rejection Adaptive Metropolis (DRAM).
- **Gradient-based samplers**: Hamiltonian Monte Carlo (with optional mass matrix), Metropolis-adjusted Langevin (MALA), and the No-U-Turn Sampler (NUTS) with dual-averaging step-size adaptation.
- **Specialized samplers**: parallel tempering (replica exchange) for multimodal targets, and preconditioned Crank-Nicolson (pCN) for high-dimensional Gaussian-prior posteriors.
- **Diagnostics**: autocorrelation, effective sample size, and Gelman-Rubin R-hat.
- **Multi-chain runner** and an extended `SampleVisualizer` (chain traces, ACF, triangular histograms, summary tables).
- **Resumable samplers**: every sampler supports `state_dict()` / `load_state_dict()` to checkpoint and resume long-running chains.

## Performance notes

The samplers are written with a "no wasted compute" discipline:

- **Gradients are evaluated exactly once per accepted move.** MALA and HMC samples carry their log-density and gradient with them, so the acceptance kernel reuses the cached value instead of triggering a second forward / backward pass through the target.
- **Log-density-only evaluations skip autograd.** Accept/reject evaluations of the target run under `torch.no_grad()`, so targets that hold `requires_grad` parameters (e.g. an `nn.Module` likelihood) do not build a graph that is never used.
- **`AdaptiveCovariance` uses a symmetric Welford update** (`M2 += ((n-1)/n) * outer(delta, delta)`) which avoids both the asymmetric round-off of the textbook two-pass form and the second outer product.
- **`AdaptiveCovariance` caches the identity matrix by default** (`cache_identity=True`); pass `cache_identity=False` if you are running in very high dimensions and prefer to recreate it each step rather than pay `dim**2` floats of persistent storage.
- **Cholesky factor recomputation can be amortised** via `adapt_interval` (default `1`, i.e. recompute every step). Larger values are useful when the `O(dim**3)` Cholesky becomes a bottleneck. The running mean and `M2` are still updated on every step regardless of `adapt_interval`.
- **HMC and NUTS adopt accepted samples by reference** (no defensive `.clone()`); a `HamiltonianSample` passed into `sampler.x` becomes the chain's state directly. Clone before assignment if you need to keep an independent reference.

## Installation

```sh
pip install mcmc-samplers          # core
pip install "mcmc-samplers[viz]"   # + matplotlib for SampleVisualizer
```

For development:

```sh
pip install -e ".[dev]"
pytest
```

## Framework

Each sampler is a (proposal, acceptance-kernel) pipeline. Samplers can be composed by mixing and matching proposals and acceptance kernels; the `Sampler` ABC keeps a consistent API across the package.

### Proposals

* `GaussianRandomWalk`: Gaussian random walk.
* `AdaptiveCovariance`: Gaussian random walk whose covariance is updated online via a numerically-stable symmetric Welford recursion. Accepts `cache_identity` and `adapt_interval` for explicit control over the memory/recompute tradeoffs (see the *Performance notes* section).
* `ScaledCovariance`: lightweight view of another proposal's covariance, scaled by a constant.
* `HamiltonianDynamics`: leapfrog-integrated Hamiltonian dynamics. Supports diagonal or full mass matrices.
* `LangevinProposal`: drift + diffusion proposal underlying MALA.
* `PreconditionedCrankNicolson`: dimension-robust symmetric proposal for Gaussian-prior posteriors.

### Samplers

* `MetropolisHastings`, `AdaptiveMetropolis`, `DelayedRejectionAdaptiveMetropolis`
* `HamiltonianMonteCarlo`, `MetropolisAdjustedLangevin`, `NUTS`
* `ParallelTempering` (replica exchange)

Every sampler accepts a `generator: torch.Generator` for fully deterministic and resumable runs.

## Example: Banana distribution

A walkthrough of the [banana example](examples/banana.ipynb).

```python
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer

class Banana:
    def __init__(self):
        self.mvn = MultivariateNormal(torch.zeros(2),
                                      torch.tensor([[1., 0.9], [0.9, 1.]]))
    def __call__(self, x):
        y = torch.stack([x[..., 0], x[..., 1] + (x[..., 0] + 1.) ** 2], dim=-1)
        return self.mvn.log_prob(y)

target = Banana()
dram = DelayedRejectionAdaptiveMetropolis(
    target=target,
    x0=torch.tensor([0., -1.]),
    cov=torch.tensor([[1., 0.9], [0.9, 1.]]),
    generator=torch.Generator().manual_seed(0),
)

samples, log_probs = dram(N=10_000, burn_in=2_000, thin=2)
visualizer = SampleVisualizer(samples)
print(visualizer.summary(labels=['x_1', 'x_2']))
```

## Multi-chain diagnostics

```python
from mcmc_samplers import run_chains, diagnostics, MetropolisHastings

def factory(chain_idx):
    return MetropolisHastings(
        target=target,
        x0=torch.randn(2),
        cov=torch.eye(2),
        generator=torch.Generator().manual_seed(chain_idx),
    )

chains, log_probs = run_chains(factory, num_chains=4, N=4_000, burn_in=500)
print('R-hat:', diagnostics.gelman_rubin(chains))
print('ESS  :', diagnostics.effective_sample_size(chains.reshape(-1, 2)))
```

## Resuming a chain

```python
sampler = DelayedRejectionAdaptiveMetropolis(target=target, x0=x0, cov=cov,
                                             generator=torch.Generator().manual_seed(0))
samples_a, _ = sampler(5_000)
state = sampler.state_dict()

# ... later, possibly in a new process ...
sampler = DelayedRejectionAdaptiveMetropolis(target=target, x0=x0, cov=cov,
                                             generator=torch.Generator())
sampler.load_state_dict(state)
samples_b, _ = sampler(5_000)  # picks up where the previous run left off
```

## More examples

The [examples](examples/) directory contains notebooks covering:

- [Banana with DRAM and HMC](examples/banana.ipynb)
- [MALA](examples/mala.ipynb)
- [NUTS with dual-averaging](examples/nuts.ipynb)
- [Parallel tempering on a bimodal target](examples/parallel_tempering.ipynb)
- [pCN in high dimensions](examples/pcn.ipynb)
- [Multi-chain convergence diagnostics](examples/diagnostics.ipynb)

## Citation / references

- Haario, H., Saksman, E., Tamminen, J. (2001). "An adaptive Metropolis algorithm." *Bernoulli* 7: 223-242.
- Green, P.J., Mira, A. (2001). "Delayed rejection in reversible jump Metropolis-Hastings." *Biometrika* 88(4): 1035-1053.
- Haario, H., Laine, M., Mira, A., Saksman, E. (2006). "DRAM: efficient adaptive MCMC." *Statistics and Computing* 16: 339-354.
- Neal, R.M. (2011). "MCMC using Hamiltonian dynamics." In *Handbook of Markov chain Monte Carlo* 2.11: 2.
- Hoffman, M.D., Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR* 15: 1593-1623.
- Roberts, G.O., Tweedie, R.L. (1996). "Exponential convergence of Langevin distributions and their discrete approximations." *Bernoulli* 2: 341-363.
- Cotter, S.L., Roberts, G.O., Stuart, A.M., White, D. (2013). "MCMC methods for functions: modifying old algorithms to make them faster." *Statistical Science* 28(3): 424-446.

## Author information

Author: Nicholas Galioto  
Email: [ngalioto@umich.edu](mailto:ngalioto@umich.edu)  
License: MIT  
Copyright 2025, Nicholas Galioto
