import numpy as np

def admm_qp(P, q, A, b, G, h, rho=1.0, max_iter=1000, tol=1e-4):
    n = P.shape[0]
    m = A.shape[0]
    p = G.shape[0]
    
    # Initialize variables
    x = np.zeros(n)
    z = np.zeros(n)
    y = np.zeros(n)
    
    # Precompute some matrices
    P_rhoI = P + rho * np.eye(n)
    P_rhoI_inv = np.linalg.inv(P_rhoI)
    
    for k in range(max_iter):
        print(f"k = {k}")

        # Update x
        x = P_rhoI_inv @ (rho * (z - y) - q)
        
        # Update z with constraints
        z_prev = z
        z = x + y
        
        # Project z onto the feasible set (Ax = b, Gx <= h)
        # This step can be complex and may require a separate QP solver for the projection
        # Here we use a simple projection onto the equality constraints
        z = z - A.T @ np.linalg.inv(A @ A.T) @ (A @ z - b)
        
        # Update y
        y = y + x - z
        
        # Check for convergence
        primal_residual = np.linalg.norm(x - z)
        dual_residual = rho * np.linalg.norm(z - z_prev)
        
        if primal_residual < tol and dual_residual < tol:
            break
    
    return x

# Example usage
P = np.array([[4, 1], [1, 2]])
q = np.array([1, 1])
A = np.array([[1, 1]])
b = np.array([1])
G = np.array([[-1, 0], [0, -1]])
h = np.array([0, 0])

x_opt = admm_qp(P, q, A, b, G, h)
print("Optimal solution:", x_opt)