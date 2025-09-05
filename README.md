# equintk

*Neural Tangent Kernel (NTK) and Neural Network Gaussian Process (NNGP) for Equinox modules.*

`equintk` provides efficient implementations of Neural Tangent Kernel computations and NTK-based predictions for neural networks built with Equinox. It includes both exact NTK computations and memory-efficient Monte Carlo approximations.

## Installation

```bash
pip install .
```

## Features

- **Exact Neural Tangent Kernel (NTK)**: Compute the exact empirical NTK with memory-efficient batching
- **Monte Carlo NTK**: Memory-efficient approximation using random projections
- **NTK Predictions**: Predict network behavior at any training time using NTK dynamics
- **Neural Network Gaussian Process (NNGP)**: Infinite-width network behavior
- **Multi-output Support**: Handle models with multiple output dimensions (e.g., RGB images)

## Usage

### Basic Neural Tangent Kernel (NTK)

```python
import jax.random as jr
import equinox as eqx
import equintk as nt

# Create a model
key = jr.PRNGKey(0)
model = eqx.nn.MLP(in_size=10, out_size=1, width_size=64, depth=2, key=key)

# Create some data
x1 = jr.normal(key, (5, 10))
x2 = jr.normal(key, (7, 10))

# Compute the exact NTK
ntk_kernel = nt.ntk(model, x1, x2, batch_size=32)
print(ntk_kernel.shape)  # (5, 7)
```

### Monte Carlo Neural Tangent Kernel

For large models where exact NTK computation is memory-prohibitive, use the Monte Carlo approximation:

```python
# Compute NTK using Monte Carlo approximation
key, subkey = jr.split(key)
mc_ntk_kernel = nt.mc_ntk(model, subkey, x1, x2, proj_dim=1000)
print(mc_ntk_kernel.shape)  # (5, 7)

# Higher proj_dim gives better approximation but uses more memory
# Typical values: 100-2000 depending on your memory constraints
```

### NTK Predictions

Predict how a neural network will behave during training using NTK dynamics:

```python
# Training data
x_train = jr.normal(key, (100, 10))
y_train = jr.normal(key, (100, 1))

# Test data
x_test = jr.normal(key, (20, 10))

# Predict network output at training time t=1.0
y_pred = nt.ntk_predict(model, x_train, y_train, x_test, t=1.0)
print(y_pred.shape)  # (20, 1)

# Predict at multiple time points
times = [0.1, 0.5, 1.0, 2.0, 5.0]
y_pred_times = nt.ntk_predict(model, x_train, y_train, x_test, t=times)
print(y_pred_times.shape)  # (5, 20, 1)
```

### Monte Carlo NTK Predictions

For memory-efficient predictions with large models:

```python
# Use Monte Carlo NTK for predictions
key, subkey = jr.split(key)
mc_y_pred = nt.mc_ntk_predict(
    model, subkey, x_train, y_train, x_test, 
    t=1.0, proj_dim=1000, ridge=1e-6
)
print(mc_y_pred.shape)  # (20, 1)
```

### Multi-Output Models

`equintk` handles multi-output models (e.g., for image reconstruction, RGB prediction):

```python
# Create a model with 3 outputs (e.g., RGB)
model = eqx.nn.MLP(in_size=2, out_size=3, width_size=64, depth=4, key=key)

# Input coordinates (e.g., pixel positions)
x_coords = jr.normal(key, (256, 2))  # 16x16 image flattened
y_rgb = jr.normal(key, (256, 3))     # RGB values

# Predict RGB values at new coordinates
rgb_pred = nt.ntk_predict(model, x_coords, y_rgb, x_coords, t=1.0)
print(rgb_pred.shape)  # (256, 3)

# Reshape back to image
rgb_image = rgb_pred.reshape(16, 16, 3)
```

### Neural Network Gaussian Process (NNGP)

Compute the infinite-width limit behavior:

```python
# Define model architecture as a function
model_fn = lambda key: eqx.nn.MLP(in_size=10, out_size=1, width_size=64, depth=2, key=key)

# Compute NNGP kernel
nngp_kernel = nt.nngp(model_fn, key, x1, x2)
print(nngp_kernel.shape)  # (5, 7)
```

## Performance Tips

### Memory Management

- Use `batch_size` parameter in `ntk()` to control memory usage
- For very large models, prefer `mc_ntk()` with appropriate `proj_dim`
- Start with `proj_dim=500-1000` and increase if you need better approximation

### Choosing Between Exact and Monte Carlo NTK

- **Exact NTK**: Use when memory allows. Provides exact results.
- **Monte Carlo NTK**: Use for large models. Provides good approximation with much less memory.

### Typical Projection Dimensions

- `proj_dim=100-500`: Fast but less accurate
- `proj_dim=1000-2000`: Good balance of speed and accuracy  
- `proj_dim=5000+`: High accuracy but slower

## Example: Image Reconstruction

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Load and prepare image
img = plt.imread("image.png")[..., :3]  # RGB only
img_small = jax.image.resize(img, (16, 16, 3), method="linear")

# Create coordinate grid
coords = jnp.stack(jnp.meshgrid(
    jnp.linspace(-1, 1, 16),
    jnp.linspace(-1, 1, 16),
    indexing="ij"
), axis=-1).reshape(-1, 2)
pixels = img_small.reshape(-1, 3)

# Create SIREN-like network for image fitting
model = eqx.nn.MLP(in_size=2, out_size=3, width_size=64, depth=4, key=key)

# Predict using NTK dynamics
reconstructed = nt.ntk_predict(model, coords, pixels, coords, t=1.0)

# Display result
plt.imshow(reconstructed.reshape(16, 16, 3))
plt.show()
```

## API Reference

### Core Functions

- `ntk(model, x1, x2=None, batch_size=32)`: Compute exact NTK
- `mc_ntk(model, key, x1, x2=None, proj_dim=100)`: Monte Carlo NTK approximation
- `ntk_predict(model, x_train, y_train, x_test, t, batch_size=32, ridge=1e-6)`: NTK-based predictions
- `mc_ntk_predict(model, key, x_train, y_train, x_test, t, proj_dim=100, ridge=1e-6)`: MC NTK predictions
- `nngp(model_fn, key, x1, x2=None)`: Neural Network Gaussian Process

### Parameters

- `model`: Equinox neural network model
- `key`: JAX random key (for stochastic methods)
- `x1`, `x2`: Input data arrays
- `t`: Training time (float or array of floats)
- `proj_dim`: Number of random projections for Monte Carlo methods
- `batch_size`: Batch size for memory-efficient computation
- `ridge`: Ridge regularization for numerical stability

## Requirements

- JAX
- Equinox
- jaxtyping
