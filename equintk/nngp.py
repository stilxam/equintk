
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

def nngp(
    model: eqx.Module,
    key: PRNGKeyArray,
    x1: Float[Array, "batch1 *dims"],
    x2: Optional[Float[Array, "batch2 *dims"]] = None,
    samps: int = 100,
) -> Float[Array, "batch1 batch2"]:
    """
    Computes the Neural Network Gaussian Process (NNGP) kernel of an Equinox model.

    The NNGP kernel is the expected value of the dot product of the model's 
    outputs for two different inputs, under a random distribution of the 
    model's parameters.

    Args:
        model: The Equinox model (e.g., `eqx.nn.MLP`).
        key: A JAX random key.
        x1: A batch of input data with shape `(batch1, *dims)`.
        x2: An optional second batch of input data with shape `(batch2, *dims)`.
            If not provided, the kernel of `x1` with itself is computed.
        samps: The number of random samples of the model to use for the expectation.

    Returns:
        The empirical NNGP matrix of shape `(batch1, batch2)`.
    """

    def get_output(model, x):
        return model(x)

    if x2 is None:
        x2 = x1

    def sample_once(key):
        pkey, mkey = jax.random.split(key)
        new_model = model(key=mkey)

        f1 = jax.vmap(lambda x: get_output(new_model, x))(x1)
        f2 = jax.vmap(lambda x: get_output(new_model, x))(x2)

        return f1 @ f2.T

    keys = jax.random.split(key, samps)
    return jnp.mean(jax.vmap(sample_once)(keys), axis=0)
