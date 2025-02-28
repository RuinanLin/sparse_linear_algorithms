import numpy as np
from scipy.io import mmread
# from pcg import pcg
from scipy.sparse.linalg import cg

def load_matrix(filename):
    return mmread(filename).toarray()

def load_vector(filename):
    return np.loadtxt(filename)

class OSQP:
    def __init__(self, P, q, A, l, u, rho=1.0, sigma=1e-6, max_iter=100000, eps_abs=1e-3, eps_rel=1e-3):
        self.P = P  # Quadratic term (n x n matrix)
        self.q = q  # Linear term (n-dimensional vector)
        self.A = A  # Constraint matrix (m x n)
        self.l = l  # Lower bounds (m-dimensional vector)
        self.u = u  # Upper bounds (m-dimensional vector)
        self.rho = rho  # Step size for ADMM
        self.sigma = sigma  # Regularization parameter
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        self.n = P.shape[0]
        self.m = A.shape[0]

        # Initialize variables
        self.x = np.zeros(self.n)
        self.z_tilde = np.zeros(self.m)
        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)

        # Precompute matrices
        self._update_kkt()
    
    def _update_kkt(self):
        self.KKT = self.P + self.sigma * np.eye(self.n) + self.rho * self.A.T @ self.A
    
    def _update_x(self):
        rhs = self.sigma * self.x - self.q + self.A.T @ (self.rho * self.z - self.y)
        # self.x = pcg(self.KKT, rhs)
        self.x, info = cg(self.KKT, rhs)
        print(f"info = {info}")
    
    def _update_z_tilde(self):
        self.z_tilde = self.A @ self.x

    def _update_z(self):
        self.z = np.clip(self.z_tilde + 1.0/self.rho * self.y, self.l, self.u)

    def _update_y(self):
        self.y += self.rho * (self.z_tilde - self.z)
    
    def _check_convergence(self):
        r_prim = self.A @ self.x - self.z
        r_dual = self.P @ self.x + self.q + self.A.T @ self.y
        r_prim_norm = np.linalg.norm(r_prim, ord=np.inf)
        print(f"r_prim_norm = {r_prim_norm}")
        r_dual_norm = np.linalg.norm(r_dual, ord=np.inf)
        print(f"r_dual_norm = {r_dual_norm}")
        
        # self.rho *= np.sqrt(r_prim_norm / r_dual_norm)
        # print(f"rho = {self.rho}")
        if r_prim_norm > 10 * r_dual_norm:
            self.rho *= 2
            print(f"rho *= 2, rho = {self.rho}")
        elif r_dual_norm > 10 * r_prim_norm:
            self.rho /= 2
            print(f"rho /= 2, rho = {self.rho}")

        self._update_kkt()
        eps_prim = self.eps_abs + self.eps_rel * max(np.linalg.norm(self.A @ self.x, ord=np.inf), np.linalg.norm(self.z, ord=np.inf))
        eps_dual = self.eps_abs + self.eps_rel * max(np.linalg.norm(self.P @ self.x, ord=np.inf), np.linalg.norm(self.A.T @ self.y, ord=np.inf), np.linalg.norm(self.q, ord=np.inf))
        return r_prim_norm < eps_prim and r_dual_norm < eps_dual
    
    def solve(self):
        """Run the ADMM algorithm."""
        for i in range(self.max_iter):
            print(f"i = {i}")
            self._update_x()
            self._update_z_tilde()
            self._update_z()
            self._update_y()

            if self._check_convergence():
                break
        return self.x

def main():
    P = load_matrix("P.mtx")
    q = load_vector("q.txt")
    A = load_matrix("A.mtx")
    l = load_vector("l.txt")
    u = load_vector("u.txt")

    solver = OSQP(P, q, A, l, u)
    x_opt = solver.solve()
    print("Optimal solution:", x_opt)

if __name__ == '__main__':
    main()