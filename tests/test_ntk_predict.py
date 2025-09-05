import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from equintk.ntk import ntk, mc_ntk
from equintk.ntk_predict import ntk_predict, mc_ntk_predict


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
def get_data():
    key = jax.random.PRNGKey(0)
    model_key, train_key, test_key = jax.random.split(key, 3)
    
    model = MLP(model_key)
    
    x_train = jax.random.normal(train_key, (10, 2))
    y_train = jax.random.normal(train_key, (10, 1))
    x_test = jax.random.normal(test_key, (5, 2))
    
    return model, x_train, y_train, x_test

def test_ntk_predict_t0(get_data):
    """Test that at t=0, the prediction is the initial model prediction."""
    model, x_train, y_train, x_test = get_data
    
    y_pred_t0 = ntk_predict(model, x_train, y_train, x_test, t=0.0)
    
    y_initial = jax.vmap(model)(x_test)
    
    assert jnp.allclose(y_pred_t0, y_initial, atol=1e-6)

def test_ntk_predict_t_inf(get_data):
    """Test that as t -> inf, the prediction converges to the kernel regression solution."""
    model, x_train, y_train, x_test = get_data
    
    y_pred_t_inf = ntk_predict(model, x_train, y_train, x_test, t=1e6)
    
    k_test_train = ntk(model, x_test, x_train)
    k_train_train = ntk(model, x_train, None)
    f0_train = jax.vmap(model)(x_train)
    f0_test = jax.vmap(model)(x_test)
    
    k_inv_r0 = jnp.linalg.solve(k_train_train + 1e-6 * jnp.eye(k_train_train.shape[0]), y_train - f0_train)
    y_expected = f0_test + k_test_train @ k_inv_r0
    
    assert jnp.allclose(y_pred_t_inf, y_expected, atol=1e-4)

def test_ntk_predict_t_array(get_data):
    """Test that giving an array of times returns predictions for each time."""
    model, x_train, y_train, x_test = get_data
    
    times = jnp.array([0.0, 1.0, 10.0])
    
    y_preds = ntk_predict(model, x_train, y_train, x_test, t=times)
    
    assert y_preds.shape == (len(times), x_test.shape[0], y_train.shape[1])
    
    y_initial = jax.vmap(model)(x_test)
    assert jnp.allclose(y_preds[0], y_initial, atol=1e-6)


def test_mc_ntk_predict_t0(get_data):
    """Test that at t=0, the mc prediction is the initial model prediction."""
    model, x_train, y_train, x_test = get_data
    key = jax.random.PRNGKey(42)

    y_pred_t0 = mc_ntk_predict(model, key, x_train, y_train, x_test, t=0.0)

    y_initial = jax.vmap(model)(x_test)

    assert jnp.allclose(y_pred_t0, y_initial, atol=1e-6)


def test_mc_ntk_predict_t_inf(get_data):
    """Test that as t -> inf, the mc_ntk_predict converges to the kernel regression solution."""
    model, x_train, y_train, x_test = get_data
    key = jax.random.PRNGKey(42)

    y_pred_t_inf = mc_ntk_predict(model, key, x_train, y_train, x_test, t=1e6, proj_dim=2000)

    key1, key2 = jax.random.split(key)
    k_test_train = mc_ntk(model, key1, x_test, x_train, proj_dim=2000)
    k_train_train = mc_ntk(model, key2, x_train, None, proj_dim=2000)

    f0_train = jax.vmap(model)(x_train)
    f0_test = jax.vmap(model)(x_test)

    ridge = 1e-6
    k_inv_r0 = jnp.linalg.solve(k_train_train + ridge * jnp.eye(k_train_train.shape[0]), y_train - f0_train)
    y_expected = f0_test + k_test_train @ k_inv_r0

    assert jnp.allclose(y_pred_t_inf, y_expected, atol=1e-2, rtol=1e-2)


def test_mc_ntk_predict_t_array(get_data):
    """Test that giving an array of times returns predictions for each time for mc_ntk_predict."""
    model, x_train, y_train, x_test = get_data
    key = jax.random.PRNGKey(42)

    times = jnp.array([0.0, 1.0, 10.0])

    y_preds = mc_ntk_predict(model, key, x_train, y_train, x_test, t=times)

    assert y_preds.shape == (len(times), x_test.shape[0], y_train.shape[1])

    y_initial = jax.vmap(model)(x_test)
    assert jnp.allclose(y_preds[0], y_initial, atol=1e-6)
