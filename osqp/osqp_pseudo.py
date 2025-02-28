def osqp(P, q, A, l, u, max_iter=1000, rho=1.0, sigma=1e-6, eps_abs=1e-3, eps_rel=1e-3, mu=10, tau=2):
    n = P.shape[0]
    m = A.shape[0]
    
    x = np.zeros(n)
    z_tilde = np.zeros(m)
    z = np.zeros(m)
    y = np.zeros(m)
    
    for i in range(max_iter):
        rhs = sigma * x - q + A.T @ (rho * z - y)
        x = pcg(P + sigma * np.eye(n) + rho * A.T @ A, rhs)
        
        z_tilde = A @ x
        z = np.clip(z_tilde + 1.0/rho * y, l, u)
        
        y += rho * (z_tilde - z)

        Px = P @ x
        ATy = A.T @ y

        r_prim = z_tilde - z
        r_dual = Px + q + ATy
        r_prim_norm = np.linalg.norm(r_prim, ord=np.inf)
        r_dual_norm = np.linalg.norm(r_dual, ord=np.inf)

        if r_prim_norm > mu * r_dual_norm:
            rho *= tau
        elif r_dual_norm > mu * r_prim_norm:
            rho /= tau

        eps_prim = eps_abs + eps_rel * max(np.linalg.norm(z_tilde, ord=np.inf), np.linalg.norm(z, ord=np.inf))
        eps_dual = eps_abs + eps_rel * max(np.linalg.norm(Px, ord=np.inf), np.linalg.norm(ATy, ord=np.inf), np.linalg.norm(q, ord=np.inf))
        if r_prim_norm < eps_prim and r_dual_norm < eps_dual:
            break
    
    return x
