import numpy as np
from scipy.io import mmread
from scipy.sparse import diags

def conjugate_gradient(A, b, tol=1e-8, max_iter=10000):
    """
    Solve Ax = b using the Conjugate Gradient (CG) method.
    """
    x = np.zeros_like(b)    # Initial guess
    r = b - A @ x
    p = r.copy()
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap

        if np.linalg.norm(r_new) < tol:
            break
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p

        r = r_new

        print(f"iter {i} finished")

def preconditioned_conjugate_gradient(A, b, tol=1e-8, max_iter=1000):
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

        print(f"iter {i} finished")

    return x

def main():
    # Load matrix from .mtx file
    A = mmread("bcsstk06.mtx").tocsr()
    b = 10000 * np.ones(A.shape[0]) # Example right-hand side vector

    print("Conjugate Gradient")
    x = conjugate_gradient(A, b)

    print("Preconditioned Conjugate Gradient")
    x = preconditioned_conjugate_gradient(A, b)
    # print("x = ", x)
    # print("b - A @ x = ", b - A @ x)

if __name__ == "__main__":
    main()