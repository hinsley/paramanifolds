import jax.numpy as jnp


def normalize(v):
    return v / jnp.linalg.norm(v)