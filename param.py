import jax.numpy as jnp

def x(U): # U -> S
    u, v = U[0], U[1]

    ##### Input a parametrization (U -> S) of the surface as a 3-list.
    position = jnp.array([
        u, # x_1
        v, # x_2
        u * v, # x_3
    ])
    return position
    #####

def x_inverse(S): # S -> U
    x_1, x_2, x_3 = S[0], S[1], S[2]

    ##### Input a coordinate mapping (S -> U) on the surface as a 2-list.
    coordinates = jnp.array([
        x_1, # u
        x_2, # v
    ])
    ###
    return coordinates