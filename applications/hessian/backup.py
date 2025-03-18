    def compute_d_u_d_theta_F(F_fn, u, theta, lamda, theta_hat):
        # [(∂/∂u_k)(∂/∂θ_j)F_i] * λ_i * θ_hat_j
        # Green term

        theta_tangent = theta_hat
        f_cotangent = lamda

        def F_jvp_theta(u, theta):
            # Compute JVP of F_fn with respect to θ at (u, θ) along θ_tangent
            primals = (u, theta)  # Point for evaluation
            tangents = (jax.tree_map(np.zeros_like, u), theta_tangent)  # Zero tangent for u, θ_tangent for θ
            primals_out, F_jvp = jax.jvp(F_fn, primals, tangents) # F(u, θ), F_jvp = F_jvp_u + F_jvp_θ
            return F_jvp

        # Compute VJP of F_jvp_theta with respect to u at (u, θ) along f_cotangent
        primals_out, vjp_func = jax.vjp(F_jvp_theta, u, theta)
        vjp_u, vjp_theta = vjp_func(f_cotangent) # vjp_u, vjp_θ

        return vjp_u

    def compute_d_theta_d_u_F(F_fn, u, theta, lamda, u_hat):
        # [(∂/∂θ_k)(∂/∂u_j)F_i] * λ_i * u_hat_j
        # Pink term and orange term

        u_tangent = u_hat

        def sum_jvp(u, theta):
            # Compute JVP of F_fn with respect to u at (u, θ) along u_tangent
            primals = (u, theta)  # Point for evaluation
            tangents = (u_tangent, jax.tree_map(np.zeros_like, theta))  # u_tangent for u, zero tangent for θ 
            primals_out, F_jvp = jax.jvp(F_fn, primals, tangents) # F(u, θ), F_jvp = F_jvp_u + F_jvp_θ
            result = jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_map(lambda x, y: np.sum(x*y), primals_out, lamda))
            return result

        d_theta_d_u_F = jax.grad(sum_jvp, argnums=1)(u, theta) # Pink term
        d_u_d_u_F = jax.grad(sum_jvp, argnums=0)(u, theta) # Orange term

        return d_theta_d_u_F, d_u_d_u_F