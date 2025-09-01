import jax
import jax.numpy as jnp
import equinox as eqx
from equintk.ntk import ntk, _ntk_fn_single_batch

# Create a simple test case
key = jax.random.PRNGKey(0)
mkey, dkey = jax.random.split(key)

# Small test case for easier debugging
in_dim = 3
out_dim = 2
n_samples = 4
batch_size = 2

model = eqx.nn.MLP(in_size=in_dim, out_size=out_dim, width_size=4, depth=1, key=mkey)
x = jax.random.normal(dkey, (n_samples, in_dim))

print(f"Model output shape for first sample: {model(x[0]).shape}")

# Test the core function directly
print("\n=== Testing _ntk_fn_single_batch directly ===")
kernel_core_symmetric = _ntk_fn_single_batch(model, x, x2=None)
kernel_core_nonsymmetric = _ntk_fn_single_batch(model, x, x2=x)

print(f"Core symmetric kernel:\n{kernel_core_symmetric}")
print(f"Core non-symmetric kernel:\n{kernel_core_nonsymmetric}")
print(f"Core difference:\n{kernel_core_symmetric - kernel_core_nonsymmetric}")

# Test both implementations
print("\n=== Testing full ntk function ===")
kernel_symmetric = ntk(model, x, x2=None, batch_size=batch_size)
kernel_brute_force = ntk(model, x, x, batch_size=batch_size)

print(f"Symmetric kernel:\n{kernel_symmetric}")
print(f"Brute force kernel:\n{kernel_brute_force}")
print(f"Difference:\n{kernel_symmetric - kernel_brute_force}")

# Test if they're symmetric
print(f"Symmetric kernel is symmetric: {jnp.allclose(kernel_symmetric, kernel_symmetric.T)}")
print(f"Brute force kernel is symmetric: {jnp.allclose(kernel_brute_force, kernel_brute_force.T)}")

# Test with smaller batches to see individual block computations
print("\n=== Testing with batch_size=1 ===")
kernel_symmetric_bs1 = ntk(model, x, x2=None, batch_size=1)
kernel_brute_force_bs1 = ntk(model, x, x, batch_size=1)

print(f"Symmetric kernel (bs=1):\n{kernel_symmetric_bs1}")
print(f"Brute force kernel (bs=1):\n{kernel_brute_force_bs1}")
print(f"Difference (bs=1):\n{kernel_symmetric_bs1 - kernel_brute_force_bs1}")
