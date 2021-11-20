import jax.numpy as jnp

from fundamental_forms import Dx, ffI, ffII


shape = lambda u, v: ffII(u, v) @ jnp.linalg.inv(ffI(u, v))

principal_curvature = lambda u, v: jnp.linalg.eigvals(shape(u, v))
# Principal directions in the domain of the parametrization (the chart)
principal_direction = lambda u, v: jnp.linalg.eig(shape(u, v))[1]

# Gaussian curvature
K = lambda u, v: jnp.product(principal_curvature(u, v))

# Principal directions in the image of the parametrization are obtained via the
# tangent map Dx : T_(u, v) U -> T_x(u, v) R^3.
principal_direction_image = lambda u, v: Dx(u, v) @ principal_direction(u, v)

# Careful, the principal directions are the *columns* here, not the rows! Make sure
# to take the transpose before selecting rows for direction vectors.
