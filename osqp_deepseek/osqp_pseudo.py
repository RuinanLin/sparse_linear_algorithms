def osqp(P, q, A, l, u, rho=1.0, rho_min=1e-6, rho_max=1e6, mu=10, tau=1.5, adjust_interval=25, max_iter=1000, tol=1e-3, _pcg_max_iter=1000, _pcg_tol=1e-5):
    n = q.shape[0]
    m = A.shape[0]

    x = np.zeros(n)
    z = np.zeros(m)
    u_dual = np.zeros(m)
    z_prev = z.copy()
    
    diag_P = np.diag(P)
    A_sq_sum = (A ** 2).sum(axis=0)

    diag_K = diag_P + rho * A_sq_sum
    M_diag = 1.0 / diag_K
    
    for iter in range(max_iter):
        rhs = -q + rho * (A.T @ (z - u_dual))

        x = np.zeros(n)
        _pcg_r = rhs
        _pcg_z = _pcg_r * M_diag
        _pcg_p = _pcg_z.copy()
        _pcg_rsold = np.dot(_pcg_r, _pcg_z)

        for _pcg_i in range(_pcg_max_iter):
            _pcg_Ap = P @ _pcg_p + rho * (A.T @ (A @ _pcg_p))
            _pcg_alpha = _pcg_rsold / np.dot(_pcg_p, _pcg_Ap)
            x += _pcg_alpha * _pcg_p
            _pcg_r -= _pcg_alpha * _pcg_Ap
            _pcg_z = _pcg_r * M_diag
            _pcg_rsnew = np.dot(_pcg_r, _pcg_z)

            if np.sqrt(_pcg_rsnew) < _pcg_tol:
                break
            
            _pcg_beta = _pcg_rsnew / _pcg_rsold
            _pcg_p = _pcg_z + _pcg_beta * _pcg_p
            _pcg_rsold = _pcg_rsnew
        
        Ax = A @ x
        Ax_plus_u = Ax + u_dual
        z = np.clip(Ax_plus_u, l, u)

        u_dual += Ax - z

        if iter % adjust_interval == 0 and iter > 0:
            r_norm = np.linalg.norm(Ax - z)
            s_norm = np.linalg.norm(rho * A.T @ (z - z_prev))

            if r_norm > mu * s_norm:
                new_rho = rho * tau
            elif s_norm > mu * r_norm:
                new_rho = rho / tau
            else:
                new_rho = rho
            
            new_rho = np.clip(new_rho, rho_min, rho_max)
            if new_rho != self.rho:
                rho = new_rho
            
            z_prev = z.copy()
        
        primal_res = np.linalg.norm(Ax - z)
        dual_res = np.linalg.norm(rho * A.T @ (z - z_prev))
        if primal_res < tol and dual_res < tol:
            break
        
    return x