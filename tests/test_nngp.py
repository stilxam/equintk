
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from equintk.nngp import nngp
from equintk.ntk import ntk

# Setup a simple MLP for testing
class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(2, 8, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(8, 1, key=key2),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@pytest.fixture
def get_model_and_data():
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    # model needs to be a function that accepts a key
    model = MLP
    
    x1 = jax.random.normal(data_key, (10, 2))
    x2 = jax.random.normal(data_key, (5, 2))
    
    return model, x1, x2

def test_shape_correctness(get_model_and_data):
    """Verify that nngp(model, key, x1, x2) returns a matrix of the expected shape (batch1 x batch2)."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel = nngp(model, key, x1, x2)
    assert kernel.shape == (x1.shape[0], x2.shape[0])

def test_symmetry(get_model_and_data):
    """Check that nngp(model, key, x, x) produces a symmetric matrix."""
    model, x1, _ = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel = nngp(model, key, x1, x1)
    assert jnp.allclose(kernel, kernel.T, atol=1e-6)

def test_equivalence(get_model_and_data):
    """Ensure nngp(model, key, x1, x2) is equal to nngp(model, key, x2, x1).T."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel1 = nngp(model, key, x1, x2)
    kernel2 = nngp(model, key, x2, x1)
    assert jnp.allclose(kernel1, kernel2.T, atol=1e-6)

def test_known_value_linear_model():
    """For a trivial linear model y = w*x, the NNGP is simply x1 * x2.T."""
    key = jax.random.PRNGKey(42)
    
    # A simple linear model with one parameter
    class LinearModel(eqx.Module):
        weight: jnp.ndarray

        def __init__(self, key):
            self.weight = jax.random.normal(key, (1,))

        def __call__(self, x):
            # The model should return a scalar for the NTK calculation.
            # We'll assume the input `x` is a scalar for simplicity.
            return self.weight * x[0]

    model = LinearModel
    
    # Create some scalar data
    x1 = jax.random.normal(key, (3, 1))
    x2 = jax.random.normal(key, (4, 1))

    # The NNGP of f(x) = w*x is E[w*x1 * w*x2] = E[w^2] * x1 * x2.T
    # If w ~ N(0, 1), then E[w^2] = 1
    expected_kernel = x1 @ x2.T
    
    # Calculate the NNGP using the library
    actual_kernel = nngp(model, key, x1, x2)
    
    assert jnp.allclose(expected_kernel, actual_kernel, atol=1e-1)
