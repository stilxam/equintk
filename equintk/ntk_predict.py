import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
import jax.numpy as jnp
import jax
from equintk.ntk import ntk, mc_ntk


def ntk_predict(
        model: eqx.Module,
        x_train: Float[Array, "batch_train *dims"],
        y_train: Float[Array, "batch_train *output_dims"],
        x_test: Float[Array, "batch_test *dims"],
        t: float | Float[Array, "num_times"],
        batch_size: int = 32,
        ridge: float = 1e-6,
) -> Float[Array, "batch_test *output_dims"]:
    """
    Predicts the output of a network at training time `t` using NTK dynamics.

    This function assumes the model has a single output, or that the outputs are
    independent and the NTK is computed by summing over the output dimensions.
    The training dynamics are predicted for gradient descent with an infinitesimal
    learning rate on a mean squared error loss.

    Args:
        model: The Equinox model.
        x_train: Training data.
        y_train: Training labels.
        x_test: Test data for which to make predictions.
        t: Training time. Can be a single float or an array of time points.
        batch_size: Batch size for the NTK computation.
        ridge: A small value to add to the diagonal of the kernel matrix for
               numerical stability during inversion.

    Returns:
        The predicted outputs on `x_test` at the given time(s) `t`.
        If `t` is a float, the output shape is `(batch_test, *output_dims)`.
        If `t` is an array of shape `(num_times,)`, the output shape is
        `(num_times, batch_test, *output_dims)`.
    """
    f0_train = jax.vmap(model)(x_train)
    f0_test = jax.vmap(model)(x_test)

    y_train = jnp.reshape(y_train, f0_train.shape)

    k_test_train = ntk(model, x_test, x_train, batch_size=batch_size)
    k_train_train = ntk(model, x_train, None, batch_size=batch_size)

    k_train_train_reg = k_train_train + ridge * jnp.eye(k_train_train.shape[0])

    eigvals, eigvecs = jnp.linalg.eigh(k_train_train_reg)

    r0 = y_train - f0_train

    original_output_shape = r0.shape[1:]
    r0_flat = r0.reshape((r0.shape[0], -1))

    def predict_for_single_t(ti):
        safe_eigvals = jnp.where(eigvals < 1e-9, 1e-9, eigvals)
        time_op_diag = (1 - jnp.exp(-ti * safe_eigvals)) / safe_eigvals

        r0_proj = eigvecs.T @ r0_flat
        evolved_r0_proj = time_op_diag[:, jnp.newaxis] * r0_proj

        delta_f_flat = (k_test_train @ eigvecs) @ evolved_r0_proj

        return f0_test + delta_f_flat.reshape((-1, *original_output_shape))

    t = jnp.asarray(t)
    if t.ndim == 0:
        return predict_for_single_t(t)
    else:
        return jax.vmap(predict_for_single_t)(t)


def mc_ntk_predict(
    model: eqx.Module,
    key: PRNGKeyArray,
    x_train: Float[Array, "batch_train *dims"],
    y_train: Float[Array, "batch_train *output_dims"],
    x_test: Float[Array, "batch_test *dims"],
    t: float | Float[Array, "num_times"],
    proj_dim: int = 100,
    ridge: float = 1e-6,
) -> Float[Array, "batch_test *output_dims"]:
    """
    Predicts the output of a network at training time `t` using Monte Carlo
    approximations of NTK dynamics.

    This function assumes the model has a single output, or that the outputs are
    independent and the NTK is computed by summing over the output dimensions.
    The training dynamics are predicted for gradient descent with an infinitesimal
    learning rate on a mean squared error loss.

    Args:
        model: The Equinox model.
        key: A JAX random key.
        x_train: Training data.
        y_train: Training labels.
        x_test: Test data for which to make predictions.
        t: Training time. Can be a single float or an array of time points.
        proj_dim: The dimension of the random projection.
        ridge: A small value to add to the diagonal of the kernel matrix for
               numerical stability during inversion.

    Returns:
        The predicted outputs on `x_test` at the given time(s) `t`.
        If `t` is a float, the output shape is `(batch_test, *output_dims)`.
        If `t` is an array of shape `(num_times,)`, the output shape is
        `(num_times, batch_test, *output_dims)`.
    """
    f0_train = jax.vmap(model)(x_train)
    f0_test = jax.vmap(model)(x_test)

    y_train = jnp.reshape(y_train, f0_train.shape)

    key1, key2 = jax.random.split(key)
    k_test_train = mc_ntk(model, key1, x_test, x_train, proj_dim=proj_dim)
    k_train_train = mc_ntk(model, key2, x_train, None, proj_dim=proj_dim)

    k_train_train_reg = k_train_train + ridge * jnp.eye(k_train_train.shape[0])

    eigvals, eigvecs = jnp.linalg.eigh(k_train_train_reg)

    r0 = y_train - f0_train

    original_output_shape = r0.shape[1:]
    r0_flat = r0.reshape((r0.shape[0], -1))

    def predict_for_single_t(ti):
        safe_eigvals = jnp.where(eigvals < 1e-9, 1e-9, eigvals)
        time_op_diag = (1 - jnp.exp(-ti * safe_eigvals)) / safe_eigvals

        r0_proj = eigvecs.T @ r0_flat
        evolved_r0_proj = time_op_diag[:, jnp.newaxis] * r0_proj

        delta_f_flat = (k_test_train @ eigvecs) @ evolved_r0_proj

        return f0_test + delta_f_flat.reshape((-1, *original_output_shape))

    t = jnp.asarray(t)
    if t.ndim == 0:
        return predict_for_single_t(t)
    else:
        return jax.vmap(predict_for_single_t)(t)
