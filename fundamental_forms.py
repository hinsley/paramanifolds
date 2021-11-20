import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jit

import param
from vector_utils import normalize

#TODO: Move Jacobian & Hessian computations to a separate module.

# The Jacobian operator
jacobian = jacrev(param.x)
Dx = lambda u, v: jacobian(jnp.array([u, v]))

x_u = lambda u, v: Dx(u, v).T[0]
x_v = lambda u, v: Dx(u, v).T[1]
n   = lambda u, v: normalize(jnp.cross(x_u(u, v), x_v(u, v)))

# The Hessian operator
hessian = jacfwd(jacobian)
D2x = lambda u, v: hessian(jnp.array([u, v]))

x_uu = lambda u, v: D2x(u, v).T[0][0]
x_uv = lambda u, v: D2x(u, v).T[0][1]
x_vu = lambda u, v: D2x(u, v).T[1][0]
x_vv = lambda u, v: D2x(u, v).T[1][1]

# The first fundamental form
def _ffI(u, v):
    _x_u = x_u(u, v)
    _x_v = x_v(u, v)
    return jnp.array(
        [[_x_u @ _x_u, _x_u @ _x_v],
         [_x_v @ _x_u, _x_v @ _x_v]]
    )
ffI = jit(_ffI)

# The second fundamental form
def _ffII(u, v):
    _n = n(u, v)
    return jnp.array(
        [[x_uu(u, v) @ _n, x_uv(u, v) @ _n],
         [x_vu(u, v) @ _n, x_vv(u, v) @ _n]]
    )
ffII = jit(_ffII)
