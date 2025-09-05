import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional, Callable
from functools import partial


@eqx.filter_jit
def _ntk_fn_single_batch(
        model: eqx.Module,
        x1: Float[Array, "batch1 *dims"],
        x2: Optional[Float[Array, "batch2 *dims"]] = None,
) -> Float[Array, "batch1 batch2"]:
    """Core function to compute the NTK for a single batch using jnp.einsum."""
    params, static = eqx.partition(model, eqx.is_array)

    def forward_fn(p, x):
        return eqx.combine(p, static)(x)

    param_leaves, _ = jax.tree_util.tree_flatten(params)
    total_params = sum([jnp.size(leaf) for leaf in param_leaves])
    x1_first = jax.tree.map(lambda x: x[0], x1)
    output_dim = jnp.size(forward_fn(params, x1_first))

    jac_fn = jax.jacfwd(forward_fn) if output_dim > total_params else jax.jacrev(forward_fn)

    def get_jacobian(p, x):
        jac = jac_fn(p, x)
        leaves, _ = jax.tree_util.tree_flatten(jac)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    J_fn = jax.vmap(get_jacobian, in_axes=(None, 0), out_axes=0)

    J1 = J_fn(params, x1)
    J2 = J1 if x2 is None else J_fn(params, x2)

    return jnp.einsum('bp,cp->bc', J1, J2)


@eqx.filter_jit
def _ntk_scan(
        model: eqx.Module,
        x1: Float[Array, "batch1 *dims"],
        x2: Float[Array, "batch2 *dims"],
        batch_size_x1: int,
        batch_size_x2: int,
) -> Float[Array, "batch1 batch2"]:
    """JIT-compilable function to compute the NTK block-by-block using lax.scan."""
    n1, n2 = x1.shape[0], x2.shape[0]

    def compute_row_of_blocks(x1_batch):
        def scan_body_for_cols(carry, j):
            x2_batch = jax.lax.dynamic_slice(x2, (j, *([0] * (x2.ndim - 1))), (batch_size_x2, *x2.shape[1:]))
            return carry, _ntk_fn_single_batch(model, x1_batch, x2_batch)

        j_steps = jnp.arange(0, n2, batch_size_x2)
        _, kernel_blocks = jax.lax.scan(scan_body_for_cols, None, j_steps)
        return jnp.concatenate(kernel_blocks, axis=1)

    def scan_body_for_rows(carry, i):
        x1_batch = jax.lax.dynamic_slice(x1, (i, *([0] * (x1.ndim - 1))), (batch_size_x1, *x1.shape[1:]))
        return carry, compute_row_of_blocks(x1_batch)

    i_steps = jnp.arange(0, n1, batch_size_x1)
    _, kernel_rows = jax.lax.scan(scan_body_for_rows, None, i_steps)

    return jnp.concatenate(kernel_rows, axis=0)



def ntk(
        model: eqx.Module,
        x1: Float[Array, "batch1 *dims"],
        x2: Optional[Float[Array, "batch2 *dims"]] = None,
        batch_size: int = 32,
) -> Float[Array, "batch1 batch2"]:
    """
    Computes the exact empirical NTK using a JIT-compiled batching loop (`lax.scan`).

    This version is memory-efficient, fast, and numerically robust. It handles
    large inputs by processing them in batches on-device and explicitly enforces
    the symmetry of the kernel matrix when `x2` is not provided.

    Args:
        model: The Equinox model (e.g., `eqx.nn.MLP`).
        x1: A batch of input data.
        x2: An optional second batch of input data. If None, the kernel of `x1`
            with itself is computed.
        batch_size: The size of the micro-batches. A smaller size reduces
            memory usage at the cost of potential performance.

    Returns:
        The empirical NTK matrix of shape `(batch1, batch2)`.
    """
    is_symmetric = x2 is None
    _x2 = x1 if is_symmetric else x2
    n1_orig, n2_orig = x1.shape[0], _x2.shape[0]

    pad1 = (batch_size - (n1_orig % batch_size)) % batch_size
    pad2 = (batch_size - (n2_orig % batch_size)) % batch_size
    x1_padded = jnp.pad(x1, [(0, pad1)] + [(0, 0)] * (x1.ndim - 1))
    x2_padded = jnp.pad(_x2, [(0, pad2)] + [(0, 0)] * (_x2.ndim - 1))

    kernel_padded = _ntk_scan(model, x1_padded, x2_padded, batch_size, batch_size)

    if is_symmetric:
        kernel_padded = jnp.triu(kernel_padded) + jnp.triu(kernel_padded, k=1).T

    return kernel_padded[:n1_orig, :n2_orig]

def mc_ntk(
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

    def compute_jacobian_projections(x_batch):
        def single_jvp(v_idx):
            v_slice = jax.tree.map(lambda y: y[..., v_idx], v)
            
            jvps = jax.vmap(lambda x: jvp(params, x, v_slice))(x_batch)
            
            return jvps.reshape(jvps.shape[0], -1)

        
        jac_projs = jax.vmap(single_jvp)(jnp.arange(proj_dim))  
        return jnp.transpose(jac_projs, (1, 0, 2))  

    J1_proj = compute_jacobian_projections(x1)  

    if x2 is None:
        J2_proj = J1_proj
    else:
        J2_proj = compute_jacobian_projections(x2)  

    
    J1_flat = J1_proj.reshape(J1_proj.shape[0], -1)  
    J2_flat = J2_proj.reshape(J2_proj.shape[0], -1)  

    
    kernel = jnp.dot(J1_flat, J2_flat.T) / proj_dim

    return kernel
