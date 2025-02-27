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

    n_zero = np.zeros(n)
    m_zero = np.zeros(m)
    neg_q = -q.copy()
    
    for iter in range(max_iter):
        rhs = neg_q.copy()
        z_0 = z.copy()
        z_0 += (-1.0) * u_dual
        rhs_0 = A.T @ z_0
        rhs += rho * rhs_0

        x = n_zero.copy()
        _pcg_r = rhs.copy()
        _pcg_z = _pcg_r * M_diag
        _pcg_p = _pcg_z.copy()
        _pcg_rsold = np.dot(_pcg_r, _pcg_z)

        for _pcg_i in range(_pcg_max_iter):
            _pcg_Ap = P @ _pcg_p
            _pcg_Ap_0 = A @ _pcg_p
            _pcg_Ap_1 = A.T @ _pcg_Ap_0
            _pcg_Ap += rho * _pcg_Ap_1
            _pcg_alpha_0 = np.dot(_pcg_p, _pcg_Ap)
            _pcg_alpha = _pcg_rsold / _pcg_alpha_0
            
            x += _pcg_alpha * _pcg_p
            _pcg_r += (-_pcg_alpha) * _pcg_Ap
            _pcg_z = _pcg_r * M_diag
            _pcg_rsnew = np.dot(_pcg_r, _pcg_z)

            if np.sqrt(_pcg_rsnew) < _pcg_tol:
                break
            
            _pcg_beta = _pcg_rsnew / _pcg_rsold
            
            _pcg_p_0 = _pcg_z.copy()
            _pcg_p_0 += _pcg_beta * _pcg_p
            _pcg_p = _pcg_p_0.copy()
            
            _pcg_rsold = _pcg_rsnew
        
        Ax = A @ x
        
        Ax_plus_u = Ax.copy()
        Ax_plus_u += 1.0 * u_dual

        z = np.clip(Ax_plus_u, l, u)

        Ax_minus_z = Ax.copy()
        Ax_minus_z += (-1.0) * z
        u_dual += (1.0) * Ax_minus_z

        r_norm = np.linalg.norm(Ax_minus_z)
        if iter % adjust_interval == 0 and iter > 0:
            z_1 = z.copy()
            z_1 += (-1.0) * z_prev
            z_2 = A.T @ z_1
            z_3 = m_zero.copy()
            z_3 += rho * z_2
            s_norm = np.linalg.norm(z_3)

            if r_norm > mu * s_norm:
                rho *= tau
                rho = np.clip(rho, rho_min, rho_max)
            elif s_norm > mu * r_norm:
                rho /= tau
                rho = np.clip(rho, rho_min, rho_max)
            
            z_prev = z.copy()
        
        z_1 = z.copy()
        z_1 += (-1.0) * z_prev
        z_2 = A.T @ z_1
        z_3 = m_zero.copy()
        z_3 += rho * z_2
        dual_res = np.linalg.norm(z_3)

        if r_norm < tol and dual_res < tol:
            break

    return x