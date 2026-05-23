# Design notes: efficiency, complexity, and usability

This document records design choices made during the MCMC package overhaul. It is meant for future development — not end-user documentation. The README's *Performance notes* section is the user-facing summary; this file explains **why** things look the way they do, where we got burned, and what not to "simplify" away.

---

## Guiding principle: algorithms dictate structure

Computational methods are not ordinary software. A refactor that is "cleaner" in OOP terms can silently change the **work done per MCMC step** — and in MCMC, wasted target or gradient evaluations dominate runtime.

When reviewing changes, ask:

1. **How many times does the target get evaluated this step?** (forward-only vs forward+backward)
2. **Does this code path assume stage order?** (Delayed Rejection is sequential by definition)
3. **Does `is_symmetric` actually let us skip work, or only cancel terms in the ratio?**
4. **Is this cache trading memory for speed, and is that tradeoff user-configurable?**

Usability wins when defaults are correct for typical use; complexity is acceptable when it mirrors the math; efficiency is non-negotiable on hot paths.

---

## The evaluation budget: target and gradients

### Rule: one forward+backward pass per point that needs a gradient

Gradient-based samplers (MALA, HMC, NUTS) should never do:

```
log_prob = target(x)          # forward only
grad = autograd(log_prob)     # second forward+backward
```

when a single `log_prob_and_grad` call suffices.

**Implementation:**

- `mcmc_samplers/utils/autograd.py::log_prob_and_grad` — always builds a fresh leaf, returns detached `(log_prob, grad)`.
- **MALA `LangevinProposal.propose`** eagerly computes `(log_prob, grad)` at the proposed point and attaches both to the returned `Sample`. The reverse-direction drift in `log_prob` during acceptance needs that gradient; computing it lazily would waste a pass.
- **MALA `MetropolisAdjustedLangevin.x` setter** overrides the base setter so initialization also uses one `log_prob_and_grad` call instead of `_eval_target` followed by a separate grad fill on the first step.

### Rule: cache on the Sample, consume on accept

Accepted chain states carry whatever the next proposal needs (`log_prob`, and for gradient methods also `grad`, and for HMC also momentum-related fields on `HamiltonianSample`). The acceptance kernel reads cached values; it does not re-evaluate unless correctness requires it.

### Gotcha: noisy HMC re-evaluation

The base `Sampler._reeval_log_prob` only refreshes `log_prob`. For HMC, **`HamiltonianMonteCarlo` must override this** to call `log_prob_and_grad`; otherwise stale gradients persist after a noisy-likelihood re-eval and leapfrog integration is wrong.

### Gotcha: `torch.no_grad()` and autograd

`log_prob_and_grad` wraps work in `torch.enable_grad()` because callers (sampling loops, diagnostics) may run under `no_grad()`. Do not assume autograd is active at the call site.

---

## Delayed rejection: the biggest nuance

**Reference:** Green & Mira (2001); Mira (2001) eq. 2.4 — [PDF](https://www.dss.uniroma1.it/RePec/mtn/articoli/2001-LIX-3_4-16.pdf).

### The algorithm is inherently sequential

Within one MCMC step, DR always runs:

```
stage 1 → (reject) → stage 2 → (reject) → … → stage n
```

There is **no valid use case** for computing `α_k` without having just computed `α_{k-1}` on the same path in the same step. Tests that call `_log_alpha(..., stage=2)` in isolation must manually seed per-step state (`_fwd_log_denom = log π(x₀)`, `_fwd_log_alpha = None`) to mirror `_sample()`.

Treating DR as "call `_log_alpha` at any stage independently" led to a stateless recursive implementation that was **correct but wasteful** — it recomputed forward denominators from scratch and had exponential q-eval cost in the recursion structure. That was a software-generalization mistake, not an algorithmic one.

### Correct efficient shape: incremental forward + iterative reverse

Per MCMC step (reset in `DelayedRejection._sample`):

| State | Role |
|-------|------|
| `_fwd_log_denom` | Running log denominator on the **forward** path; starts at `log π(x₀)` |
| `_fwd_log_alpha` | Log acceptance at the previous stage on the forward path |

At stage `n`, update:

```
log D_n = log D_{n-1} + log(1 - α_{n-1}) + [intermediate q terms]
log N_n = same structure on the **reversed** path (y_n, y_{n-1}, …, y_0)
log α_n = log N_n - log D_n
```

Reverse numerators are built by looping `k = 1, …, n-1` on reversed sub-paths, threading `(rev_log_denom, rev_log_alpha)` explicitly. Inner calls reuse `_stage_log_alpha` but **do not** mutate shared instance state — only the top-level forward path uses `_fwd_*` memoization.

**Complexity:** `O(n)` proposal `log_prob` evaluations per stage attempt, matching the original hand-optimized implementation (e.g. 2 q-evals for symmetric 2-stage DRAM, 6 for fully asymmetric 2-stage).

These caches are **not serialized** in `state_dict` — they carry no information across MCMC steps.

### Gotcha: `is_symmetric` is stage-position-dependent

This tripped us up more than once.

- **Last-stage symmetric proposal:** `q_n(x_{n-1} → x_n)` and `q_n(x_n → x_{n-1})` appear on opposite sides of the ratio and **cancel exactly** → skip both q-evals.
- **Intermediate-stage symmetric proposal:** the forward and reverse paths condition on **different** states, so the q-terms **do not cancel** → must evaluate explicitly when promoting a symmetric kernel from "final" at stage `k` to "intermediate" at stage `k+1`.

A blanket "symmetric ⇒ skip `log_prob`" is wrong for DR stage ≥ 2.

### Gotcha: `log(1 - α)` needs stable arithmetic

Use `_log1m_exp` for `log(1 - exp(log_alpha))` when `log_alpha ≤ 0`. Naive `log1p(-exp(log_alpha))` is fine for moderate values; the piecewise form avoids catastrophic cancellation when `α ≈ 1`.

### Gotcha: mutable recursion state in the original DR code

The pre-refactor implementation mutated `self._denom` / `self._alpha` during recursion. That was efficient when call order was correct but fragile (wrong order, re-entrancy, serialization). The fix is **explicit threading of `(prev_log_denom, prev_log_alpha)`** in `_stage_log_alpha`, plus **instance-level forward memoization only for the outer sequential path**.

Do not " simplify" back to shared mutable recursion state without understanding the failure modes.

### Testing DR

- Always exercise stages **in order** in unit tests.
- Keep a **stateless reference oracle** (`_mira_log_alpha` in `tests/test_delayed_rejection.py`) for small stages — exponential cost is fine for `n ≤ 3`.
- Cover: 1-stage (= MH), 2-stage symmetric DRAM, 2-stage asymmetric, 3-stage mixed sym/asym.
- Optional future hardening: monkey-patch `proposal.log_prob` to assert exact q-eval counts (2, 6, …).

---

## Adaptive covariance: numerics and amortization

### Symmetric Welford update

The textbook two-pass Welford increment uses `outer(delta, delta2)` with `delta2 = x - mean_n`. Algebraically this equals `((n-1)/n) * outer(delta, delta)` with `delta = x - mean_{n-1}`.

We use the single outer-product form because:

1. **Exact symmetry** by construction — `_M2` cannot drift off-symmetric from asymmetric rank-one updates.
2. **One fewer outer product** per adaptation step.

Do not revert to the two-delta form for "clarity" without accepting the symmetry and cost tradeoff.

### Cholesky: when to recompute vs when to cache

| Operation | Default | Rationale |
|-----------|---------|-----------|
| Identity matrix `I` in `cov + eps*I` | **Cache** (`cache_identity=True`) | Tiny per-step savings, `dim²` memory — user can disable |
| Scaled empirical covariance matrix | **Do not cache** | Scalar multiply is cheap; caching doubles persistent `dim²` storage |
| Cholesky of adapted covariance | **Every step** (`adapt_interval=1`) | Responsive adaptation; stale `sqrt_cov` lets proposal scale drift between recomputes |

`adapt_interval > 1` amortizes `O(dim³)` Cholesky — only for high-dimensional bottlenecks. Running mean and `_M2` still update every step regardless.

### Cholesky jitter loop

`_cholesky_with_jitter` uses `10**ii` (not `10^ii` — that was a real bug). Escalating diagonal jitter is required for near-singular empirical covariances during early adaptation.

### `ScaledCovariance` for DRAM stage 2

Stage 2 reads stage 1's `sqrt_cov` via a property and applies `sqrt(gamma)` scaling. No duplicate adaptation logic — when stage 1 adapts, stage 2 tracks automatically. Keep this as a **view**, not a copy.

---

## Proposal `log_prob` patterns: Cholesky vs explicit covariance

Gaussian random walks store **`sqrt_cov` (lower Cholesky)** and evaluate

```python
z = solve_triangular(sqrt_cov, delta, upper=False)
return -0.5 * z.pow(2).sum()
```

This is the right default: one triangular solve per q-eval, no explicit inverse. Passing `sqrt_cov` directly from the user avoids an initial Cholesky when they already have a factor.

**When Cholesky is wasted:** if the user only ever provides `sqrt_cov` and it never changes (fixed GRW), never re-factor. Adaptive / DRAM paths must re-factor when covariance changes.

---

## Sample and state management

### `Sample` as chain state

The Markov state is more than `point`: cached `log_prob`, and for gradient methods `grad`, must travel with the state so the next iteration does not recompute.

### HMC/NUTS: adopt by reference on accept

On acceptance, `HamiltonianMonteCarlo` / `NoUTurnSampler` assign the proposal's `HamiltonianSample` directly to `self._x` — **no defensive `.clone()`** on the hot path. The proposal output is already detached; cloning every accept would allocate unnecessarily in long chains.

**Usability caveat:** if user code retains a reference to the pre-accept sample object, it may alias mutated state. Document that callers should `.clone()` before assignment if they need independence.

### `state_dict` / `load_state_dict`

- Reconstruct `self._x` with **`self._x.__class__`** so `HamiltonianSample` stays typed after resume.
- DR forward memo (`_fwd_log_denom`, `_fwd_log_alpha`) intentionally omitted — recomputed on next `_sample`.
- `AdaptiveCovariance` persists `cache_identity` and `adapt_interval`; rebuilds `_identity` on load when caching is enabled.

### Point shape

`Sampler.x` setter normalizes `point` to **1-D** `reshape(-1)`. Do not assume `point.shape[1]` exists — 1-D tensors are the contract.

---

## Architecture choices retained from the original design

Still intentional:

- **Proposals and acceptance kernels are separate** — mix-and-match without subclass explosion.
- **`Sampler` ABC is the general MCMC loop** — `_sample` iterates `(proposal, acceptance_kernel)` pairs.
- **Hierarchy preserves API** — e.g. `MetropolisHastings` → `DelayedRejection` → `Sampler`.
- **Examples are Jupyter notebooks** with plots, not just scripts.

Efficiency work stayed within this structure rather than collapsing proposals into samplers.

---

## Bugs that looked like refactors (don't repeat)

| Issue | Symptom | Root cause |
|-------|---------|------------|
| Empty jitter loop / `10^ii` | Cholesky never retried | Typo / incomplete loop body |
| Online covariance update | `_M2` blew up or went asymmetric | Naive rank-one update instead of Welford |
| HMC acceptance | Wrong energy | Used wrong momentum slot in Hamiltonian term |
| Noisy HMC | Drift after re-eval | Base `_reeval_log_prob` didn't refresh grad |
| DR recursive "cleanup" | Extra q-evals at high stage | Stateless full recomputation ignored sequential structure |
| `load_state_dict` | HMC broke after resume | Plain `Sample` reconstructed instead of subclass |
| Circular imports | Import errors | `from mcmc_samplers import *` in submodules — use relative imports |
| NUTS step-size adapt | Wrong exp / late lock | Used tensor `.exp().item()` incorrectly; lock after last warmup iter |

When "cleaning up" math-heavy code, **count evaluations** and **compare against a paper reference** before merging.

---

## Testing philosophy for numerical code

1. **Reference oracles for small cases** — recursive Mira formula for DR; closed-form MH for symmetric 1-stage; offline covariance for Welford.
2. **Invariants over mocks** — e.g. MALA step uses exactly one `log_prob_and_grad` per iteration (`tests/test_efficiency.py`).
3. **Respect algorithm call order** — DR tests seed forward memo and call stage 1 before stage 2.
4. **Symmetry tests on `_M2`** — exact symmetry after many updates, not just "close enough".

Property-based fuzzing is less valuable here than oracles tied to published equations.

---

## Explicit non-goals / deferred items

- **Memoizing full scaled covariance matrices** — default off; marginal benefit vs memory.
- **Further DR micro-optimizations** without q-eval counting evidence — current `O(n)` structure matches original; profile before touching.
- **Automatic differentiation through the entire MCMC loop** — samples detach; each step is a fresh graph. This is intentional for memory and semantics.

---

## Checklist for future algorithms

When adding a sampler or proposal:

- [ ] Count target evals (and grads) per **accepted** and **rejected** step on paper.
- [ ] If multi-stage, is work incremental across stages within one step?
- [ ] Does `is_symmetric` affect **which** q-terms are computed, not just documentation?
- [ ] Are caches optional with sensible defaults and documented memory cost?
- [ ] Does `state_dict` capture everything needed to resume **except** ephemeral per-step memo?
- [ ] Is there a notebook example?
- [ ] Is there a test with a hand-derived or literature reference acceptance ratio?

---

## Related files

| Topic | Location |
|-------|----------|
| DR implementation | `mcmc_samplers/samplers/delayed_rejection.py` |
| DR tests + Mira oracle | `tests/test_delayed_rejection.py` |
| Adaptive covariance | `mcmc_samplers/proposals/gaussian_random_walk.py` |
| MALA eager grad | `mcmc_samplers/proposals/langevin.py`, `mcmc_samplers/samplers/langevin.py` |
| Autograd helper | `mcmc_samplers/utils/autograd.py` |
| Efficiency invariants | `tests/test_efficiency.py` |
| User-facing summary | `README.md` → *Performance notes* |
