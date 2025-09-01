import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from equintk.ntk import ntk, ntk_mc

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
    model = MLP(model_key)
    
    x1 = jax.random.normal(data_key, (10, 2))
    x2 = jax.random.normal(data_key, (5, 2))
    
    return model, x1, x2

def test_shape_correctness(get_model_and_data):
    """Verify that ntk(model, x1, x2) returns a matrix of the expected shape (batch1 x batch2)."""
    model, x1, x2 = get_model_and_data
    kernel = ntk(model, x1, x2)
    assert kernel.shape == (x1.shape[0], x2.shape[0])

def test_symmetry(get_model_and_data):
    """Check that ntk(model, x, x) produces a symmetric matrix."""
    model, x1, _ = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel = ntk(model, x1, x1)
    assert jnp.allclose(kernel, kernel.T, atol=1e-6)

def test_equivalence(get_model_and_data):
    """Ensure ntk(model, x1, x2) is equal to ntk(model, x2, x1).T."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel1 = ntk(model, x1, x2)
    kernel2 = ntk(model, x2, x1)
    assert jnp.allclose(kernel1, kernel2.T, atol=1e-6)

def test_known_value_linear_model():
    """For a trivial linear model y = w*x, the NTK is simply x1 * x2.T."""
    key = jax.random.PRNGKey(42)
    
    # A simple linear model with one parameter
    class LinearModel(eqx.Module):
        weight: jnp.ndarray

        def __init__(self):
            self.weight = jnp.ones((1,))

        def __call__(self, x):
            # The model should return a scalar for the NTK calculation.
            # We'll assume the input `x` is a scalar for simplicity.
            return self.weight * x[0]

    model = LinearModel()
    
    # Create some scalar data
    x1 = jax.random.normal(key, (3, 1))
    x2 = jax.random.normal(key, (4, 1))

    # The NTK of f(x) = w*x is df/dw * df/dw = x*x
    # For two inputs, the kernel is (x1)(x2.T)
    expected_kernel = x1 @ x2.T
    
    # Calculate the NTK using the library
    actual_kernel = ntk(model, x1, x2)
    
    assert jnp.allclose(expected_kernel, actual_kernel, atol=1e-6)

def test_mc_shape_correctness(get_model_and_data):
    """Verify that ntk_mc returns a matrix of the expected shape (batch1 x batch2)."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel = ntk_mc(model, key, x1, x2)
    assert kernel.shape == (x1.shape[0], x2.shape[0])

def test_mc_symmetry(get_model_and_data):
    """Check that ntk_mc(model, x, x) produces a symmetric matrix."""
    model, x1, _ = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel = ntk_mc(model, key, x1, x1)
    assert jnp.allclose(kernel, kernel.T, atol=1e-6)

def test_mc_equivalence(get_model_and_data):
    """Ensure ntk_mc(model, x1, x2) is equal to ntk_mc(model, x2, x1).T."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    kernel1 = ntk_mc(model, key, x1, x2)
    kernel2 = ntk_mc(model, key, x2, x1)
    assert jnp.allclose(kernel1, kernel2.T, atol=1e-6)

def test_mc_approximation(get_model_and_data):
    """Check that ntk_mc is a reasonable approximation of the exact ntk."""
    model, x1, x2 = get_model_and_data
    key = jax.random.PRNGKey(0)
    
    kernel_exact = ntk(model, x1, x2)
    kernel_mc = ntk_mc(model, key, x1, x2, proj_dim=1000)
    
    assert jnp.allclose(kernel_exact, kernel_mc, atol=5e-1)

@pytest.fixture(scope="class")
def multi_output_setup():
    """
    A pytest fixture to create a multi-output model and test data.
    This setup is created once per test class, making tests faster.
    """
    key = jax.random.PRNGKey(0)
    mkey, dkey = jax.random.split(key)

    # Define parameters for the test
    in_dim = 10
    out_dim = 2  # Crucially, we are testing a multi-output scenario
    n_samples = 75  # Not a multiple of batch_size to test padding logic
    batch_size = 32

    # Create a simple MLP model with the specified output dimension
    model = eqx.nn.MLP(in_size=in_dim, out_size=out_dim, width_size=50, depth=2, key=mkey)

    # Create random input data
    x = jax.random.normal(dkey, (n_samples, in_dim))

    # Yield the setup to the test functions
    yield model, x, batch_size

def test_symmetric_multi_output(multi_output_setup):
    """
    Tests the symmetric case (x2=None) for a multi-output model.
    It confirms shape, symmetry, and correctness against the non-symmetric path.
    """
    model, x, batch_size = multi_output_setup
    n_samples = x.shape[0]

    # 1. Compute the kernel using the optimized symmetric path by setting x2=None
    kernel_symmetric = ntk(model, x, x2=None, batch_size=batch_size)

    # 2. Assert that the output has the correct shape and is symmetric
    assert kernel_symmetric.shape == (n_samples, n_samples)
    assert jnp.allclose(kernel_symmetric, kernel_symmetric.T, atol=1e-5), "Kernel must be symmetric"

    # 3. For validation, compute the kernel using the non-symmetric ("brute-force") path
    kernel_brute_force = ntk(model, x, x, batch_size=batch_size)

    # 4. Assert that the optimized result is numerically identical to the brute-force result
    assert jnp.allclose(kernel_symmetric, kernel_brute_force, atol=1e-5),\
        "The optimized symmetric implementation does not match the non-symmetric one."
