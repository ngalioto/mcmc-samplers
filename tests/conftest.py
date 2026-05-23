"""Shared pytest fixtures for the mcmc-samplers test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _set_default_dtype():
    """Use float64 throughout the tests for tighter numerical tolerances."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def seed():
    """A deterministic seed for tests that need global RNG control."""
    torch.manual_seed(0)
    return 0


def _make_generator(seed_value: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed_value)
    return g


@pytest.fixture
def make_generator():
    """Return a factory for seeded ``torch.Generator`` objects."""
    return _make_generator


class IsotropicGaussian:
    """Simple log-density for a zero-mean isotropic Gaussian."""

    def __init__(self, dim: int = 2, scale: float = 1.0):
        self.dim = dim
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (x / self.scale).pow(2).sum(dim=-1)


class CorrelatedGaussian:
    """Log-density for a correlated 2D Gaussian with known covariance."""

    def __init__(self):
        self.mean = torch.zeros(2)
        self.cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
        self.precision = torch.linalg.inv(self.cov)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        delta = x - self.mean
        return -0.5 * (delta @ self.precision * delta).sum(dim=-1)


@pytest.fixture
def isotropic_gaussian():
    return IsotropicGaussian(dim=2)


@pytest.fixture
def correlated_gaussian():
    return CorrelatedGaussian()
