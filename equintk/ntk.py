
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

def _ntk_fn(
    model: eqx.Module,
    x1: Float[Array, "batch1 *dims"],
    x2: Optional[Float[Array, "batch2 *dims"]] = None,
) -> Float[Array, "batch1 batch2"]:
    """Computes the Neural Tangent Kernel (NTK) of an Equinox model."""
    
    params, static = eqx.partition(model, eqx.is_array)

    def forward_fn(p, x):
        # Recombine to build the model for a forward pass
        return eqx.combine(p, static)(x)

    def get_jacobian(p, x):
        jac = jax.jacrev(forward_fn)(p, x)
        leaves, _ = jax.tree_util.tree_flatten(jac)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    J_fn = jax.vmap(get_jacobian, in_axes=(None, 0), out_axes=0)
    
    J1 = J_fn(params, x1)
    
    if x2 is None:
        J2 = J1
    else:
        J2 = J_fn(params, x2)

    kernel = J1 @ J2.T
    
    return kernel

_jitted_ntk = eqx.filter_jit(_ntk_fn)

def ntk(
    model: eqx.Module,
    x1: Float[Array, "batch1 *dims"],
    x2: Optional[Float[Array, "batch2 *dims"]] = None,
) -> Float[Array, "batch1 batch2"]:
    """
    Computes the Neural Tangent Kernel (NTK) of an Equinox model.

    The NTK is defined as `J(x₁)^T J(x₂)` where `J(x)` is the Jacobian of the 
    model's output with respect to its parameters. This implementation computes
    the empirical NTK for a finite-width network.

    Args:
        model: The Equinox model (e.g., `eqx.nn.MLP`).
        x1: A batch of input data with shape `(batch1, *dims)`.
        x2: An optional second batch of input data with shape `(batch2, *dims)`.
            If not provided, the kernel of `x1` with itself is computed.

    Returns:
        The empirical NTK matrix of shape `(batch1, batch2)`.
    """
    return _jitted_ntk(model, x1, x2)

def ntk_mc(
    model: eqx.Module,
    key: PRNGKeyArray,
    x1: Float[Array, "batch1 *dims"],
    x2: Optional[Float[Array, "batch2 *dims"]] = None,
    proj_dim: int = 100,
) -> Float[Array, "batch1 batch2"]:
    """
    Computes a Monte Carlo approximation of the Neural Tangent Kernel (NTK).

    This function uses random projections to approximate the NTK, which can be
    more memory-efficient for large models than computing the full Jacobian.

    Args:
        model: The Equinox model (e.g., `eqx.nn.MLP`).
        key: A JAX random key.
        x1: A batch of input data with shape `(batch1, *dims)`.
        x2: An optional second batch of input data with shape `(batch2, *dims)`.
            If not provided, the kernel of `x1` with itself is computed.
        proj_dim: The dimension of the random projection. A larger value will
            result in a more accurate approximation of the NTK.

    Returns:
        An approximation of the NTK matrix of shape `(batch1, batch2)`.
    """
    
    params, static = eqx.partition(model, eqx.is_array)
    
    def forward_fn(p, x):
        return eqx.combine(p, static)(x)

    def jvp(p, x, v):
        return jax.jvp(lambda p: forward_fn(p, x), (p,), (v,))[1]

    p_flat, p_tree = jax.tree_util.tree_flatten(params)
    p_total_size = sum([pi.size for pi in p_flat])

    key, subkey = jax.random.split(key)
    v_flat = jax.random.normal(subkey, (p_total_size, proj_dim))
    
    v_list = []
    current_index = 0
    for p in p_flat:
        v_list.append(v_flat[current_index:current_index+p.size].reshape(p.shape + (proj_dim,)))
        current_index += p.size
    v = jax.tree_util.tree_unflatten(p_tree, v_list)

    def jvp_vmap(p, x, v):
        return jax.vmap(lambda v_slice: jvp(p, x, jax.tree_util.tree_map(lambda y: y[..., v_slice], v)), in_axes=-1)(jnp.arange(proj_dim))

    J1_proj = jax.vmap(jvp_vmap, in_axes=(None, 0, None))(params, x1, v).squeeze(-1)

    if x2 is None:
        J2_proj = J1_proj
    else:
        J2_proj = jax.vmap(jvp_vmap, in_axes=(None, 0, None))(params, x2, v).squeeze(-1)

    kernel = (J1_proj @ J2_proj.T) / proj_dim
    
    return kernel
