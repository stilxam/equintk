import equinox as eqx
import jax.numpy as jnp




class PE_MLP(eqx.Module):
    mlp: eqx.Module
    num_frequencies: int

    def __init__(self, in_size, out_size, width_size, depth, num_frequencies, *, key):
        self.num_frequencies = num_frequencies
        mlp_input_size = in_size + 2 * in_size * num_frequencies
        self.mlp = eqx.nn.MLP(
            mlp_input_size,
            out_size,
            width_size,
            depth,
            key=key
        )

    def fourier_positional_encoding(self, x):

        frequencies = jnp.pi * jnp.arange(1, self.num_frequencies + 1)



        x_proj = x[..., None] * frequencies


        pe = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)



        pe_flat = pe.reshape(-1)



        return jnp.concatenate([x, pe_flat], axis=-1)

    def __call__(self, x):
        x_pe = self.fourier_positional_encoding(x)
        return self.mlp(x_pe)

