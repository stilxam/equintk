
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from equintk.nngp import nngp

# 1. Define a simple MLP model
class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(1, 64, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, 1, key=key2),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 2. Generate sine wave data
key = jax.random.PRNGKey(0)
X_train = jnp.linspace(-jnp.pi, jnp.pi, 8).reshape(-1, 1)
y_train = jnp.sin(X_train)
X_test = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)

# 3. Compute the NNGP kernel
model_key = jax.random.PRNGKey(1)
model = MLP

k_train_train = nngp(model, model_key, X_train, samps=1000)
k_test_train = nngp(model, model_key, X_test, X_train, samps=1000)
k_test_test = nngp(model, model_key, X_test, samps=1000)

# 4. Perform Gaussian Process regression
# Add a small amount of noise to the diagonal for numerical stability
k_train_train_reg = k_train_train + jnp.eye(X_train.shape[0]) * 1e-5
mean_pred = k_test_train @ jnp.linalg.solve(k_train_train_reg, y_train)
cov_pred = k_test_test - k_test_train @ jnp.linalg.solve(k_train_train_reg, k_test_train.T)
std_pred = jnp.sqrt(jnp.diag(cov_pred))

# 5. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_test, jnp.sin(X_test), 'b-', label='True sine function')
plt.plot(X_train, y_train, 'ro', label='Training data')
plt.plot(X_test, mean_pred, 'g-', label='NNGP prediction')
plt.fill_between(X_test.flatten(), 
                 (mean_pred.flatten() - 1.96 * std_pred).flatten(), 
                 (mean_pred.flatten() + 1.96 * std_pred).flatten(), 
                 color='gray', alpha=0.3, label='95% confidence interval')
plt.xlabel('x')
plt.ylabel('y')
plt.title('NNGP Regression on Sine Function')
plt.legend()
plt.grid(True)
plt.savefig('nngp_sine_plot.png')
print("Plot saved to nngp_sine_plot.png")
