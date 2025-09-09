import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import List, Callable
from jaxtyping import Array, Float, PRNGKeyArray



class SirenLayer(eqx.Module):
    weight: Float[Array, "out_features in_features"]
    bias: Float[Array, "out_features"]
    is_first: bool
    omega: float
    nonlinearity: Callable[[Float[Array, "out_features"]], Float[Array, "out_features"]] = jnp.sin

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray, is_first: bool = False, omega: float = 30.0, nonlinearity: Callable = jnp.sin):
        self.is_first = is_first
        self.omega = omega
        w_key, b_key = jr.split(key)

        if is_first:
            limit = 1 / in_features
        else:
            limit = jnp.sqrt(6 / in_features) / omega

        self.weight = jr.uniform(w_key, (out_features, in_features), minval=-limit, maxval=limit)
        self.bias = jr.uniform(b_key, (out_features,), minval=-limit, maxval=limit)
        self.nonlinearity = nonlinearity

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "out_features"]:
        activation = self.omega * (x @ self.weight.T + self.bias)
        return jnp.sin(activation)


class Siren(eqx.Module):
    layers: List[SirenLayer]

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, *, key: PRNGKeyArray):
        keys = jr.split(key, depth + 1)
        self.layers = []
        self.layers.append(SirenLayer(in_size, width_size, key=keys[0], is_first=True))
        for i in range(1, depth):
            self.layers.append(SirenLayer(width_size, width_size, key=keys[i]))
        self.layers.append(SirenLayer(width_size, out_size, key=keys[depth]))

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        for layer in self.layers:
            x = layer(x)
        return x


