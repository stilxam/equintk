# equintk

*Neural Tangent Kernel and NNGP for Equinox modules.*

## Installation

```bash
pip install .
```

## Usage

### Neural Tangent Kernel (NTK)

```python
import jax.random as jrandom
import equinox as eqx
import equintk

# Create a model
key = jrandom.PRNGKey(0)
model = eqx.nn.MLP(in_size=10, out_size=1, width_size=64, depth=2, key=key)

# Create some data
x1 = jrandom.normal(key, (5, 10))
x2 = jrandom.normal(key, (7, 10))

# Compute the NTK
ntk_kernel = equintk.ntk(model, x1, x2)

print(ntk_kernel.shape)
# (5, 7)
```

### Monte Carlo Neural Tangent Kernel (NTK-MC) (Experimental)

This feature provides a memory-efficient approximation of the NTK using Monte Carlo methods with random projections. It's particularly useful for large models where computing the full Jacobian is infeasible.

```python
import jax.random as jrandom
import equinox as eqx
import equintk

# Create a model
key = jrandom.PRNGKey(0)
model = eqx.nn.MLP(in_size=10, out_size=1, width_size=64, depth=2, key=key)

# Create some data
x1 = jrandom.normal(key, (5, 10))
x2 = jrandom.normal(key, (7, 10))

# Compute the NTK-MC
# `proj_dim` controls the trade-off between accuracy and memory.
key, subkey = jrandom.split(key)
ntk_mc_kernel = equintk.ntk_mc(model, subkey, x1, x2, proj_dim=1000)

print(ntk_mc_kernel.shape)
# (5, 7)
```

### Neural Network Gaussian Process (NNGP)

```python
import jax.random as jrandom
import equinox as eqx
import equintk

# Create a model *class*
model_fn = eqx.nn.MLP
model_args = {"in_size": 10, "out_size": 1, "width_size": 64, "depth": 2}
model = lambda key: model_fn(**model_args, key=key)


# Create some data
key = jrandom.PRNGKey(0)
x1 = jrandom.normal(key, (5, 10))
x2 = jrandom.normal(key, (7, 10))

# Compute the NNGP
nngp_kernel = equintk.nngp(model, key, x1, x2)

print(nngp_kernel.shape)
# (5, 7)
```
