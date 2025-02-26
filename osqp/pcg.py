import numpy as np

def pcg(A, b, tol=1e-8, max_iter=1000):
    """
    Solve Ax = b using the Preconditioned Conjugate Gradient (PCG) method
    with diagonal preconditioning.
    """
    x = np.zeros_like(b)    # Initial guess
    M_inv = 1.0 / A.diagonal()   # Diagonal preconditioner
    r = b - A @ x
    z = M_inv * r
    p = z.copy()

    for i in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, z) / np.dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap

        if np.linalg.norm(r_new) < tol:
            break
        
        z_new = M_inv * r_new
        beta = np.dot(r_new, z_new) / np.dot(r, z)
        p = z_new + beta * p

        r, z = r_new, z_new

    return x