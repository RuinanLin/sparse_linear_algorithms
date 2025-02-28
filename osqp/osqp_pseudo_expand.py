def osqp(P, q, A, l, u, max_iter=1000, rho=1.0, sigma=1e-6, eps_abs=1e-3, eps_rel=1e-3, mu=10, tau=2):
    n = P.shape[0]
    m = A.shape[0]
    
    x = np.zeros(n)
    z_tilde = np.zeros(m)
    z = np.zeros(m)
    y = np.zeros(m)

    P_sigma_I = P + sigma * np.eye(n)
    
    for i in range(max_iter):
        x_1 = -q
        x_1 += sigma * x
        y_1 = -y
        y_1 += rho * z
        ATy_1 = A.T @ y_1
        x_1 += 1.0 * ATy_1
        rhs = x_1

        x = pcg(P + sigma * np.eye(n) + rho * A.T @ A, rhs)

        z_tilde = A @ x

        z_tilde_1 = z_tilde
        z_tilde_1 += 1.0/rho * y
        z = np.clip(z_tilde_1, l, u)

        z_tilde_1 = z_tilde
        z_tilde_1 += -1.0 * z
        y += rho * z_tilde_1

        Px = P @ x
        ATy = A.T @ y

        r_prim = z_tilde_1
        
        r_dual = q
        r_dual += 1.0 * Px
        r_dual += 1.0 * ATy

        r_prim_norm = np.linalg.norm(r_prim, ord=np.inf)
        r_dual_norm = np.linalg.norm(r_dual, ord=np.inf)

        if r_prim_norm > mu * r_dual_norm:
            rho *= tau
        elif r_dual_norm > mu * r_prim_norm:
            rho /= tau

        z_tilde_norm = np.linalg.norm(z_tilde, ord=np.inf)
        z_norm = np.linalg.norm(z, ord=np.inf)
        Px_norm = np.linalg.norm(Px, ord=np.inf)
        eps_prim = eps_abs + eps_rel * max(z_tilde_norm, z_norm)

        ATy_norm = np.linalg.norm(ATy, ord=np.inf)
        q_norm = np.linalg.norm(q, ord=np.inf)
        eps_dual = eps_abs + eps_rel * max(Px_norm, ATy_norm, q_norm)

        if r_prim_norm < eps_prim and r_dual_norm < eps_dual:
            break

    return x
