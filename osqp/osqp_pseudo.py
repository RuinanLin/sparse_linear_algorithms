def osqp(P, q, A, l, u, max_iter=1000, rho=1.0, sigma=1e-6, eps_abs=1e-3, eps_rel=1e-3):
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

        r_prim = A @ x - z
        r_dual = P @ x + q + A.T @ y
        r_prim_norm = np.linalg.norm(r_prim, ord=np.inf)
        r_dual_norm = np.linalg.norm(r_dual, ord=np.inf)
        
        rho *= np.sqrt(r_prim_norm / r_dual_norm)

        eps_prim = eps_abs + eps_rel * max(np.linalg.norm(A @ x, ord=np.inf), np.linalg.norm(z, ord=np.inf))
        eps_dual = eps_abs + eps_rel * max(np.linalg.norm(P @ x, ord=np.inf), np.linalg.norm(A.T @ y, ord=np.inf), np.linalg.norm(q, ord=np.inf))
        if r_prim_norm < eps_prim and r_dual_norm < eps_dual:
            break
    
    return x
