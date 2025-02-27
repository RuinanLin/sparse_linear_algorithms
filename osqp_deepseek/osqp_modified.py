import numpy as np
from scipy.io import mmread

def load_matrix(filename):
    return mmread(filename).toarray()

def load_vector(filename):
    return np.loadtxt(filename)

def preconditioned_conjugate_gradient(Kx_func, b, M_diag, x0=None, tol=1e-5, max_iter=1000):
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - Kx_func(x)
    z = r * M_diag
    p = z.copy()
    rsold = np.dot(r, z)

    for i in range(max_iter):
        Ap = Kx_func(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = r * M_diag
        rsnew = np.dot(r, z)

        if np.sqrt(rsnew) < tol:
            pcg_iters = i
            break
        beta = rsnew / rsold
        p = z + beta * p
        rsold = rsnew
    return x

class OSQP:
    def __init__(self, P, q, A, l, u):
        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u
        self.n = q.shape[0]
        self.m = A.shape[0]

        self.x = np.zeros(self.n)
        self.z = np.zeros(self.m)
        self.u_dual = np.zeros(self.m)
        self.rho = 1.0
        self.rho_min = 1e-6
        self.rho_max = 1e6
        self.mu = 10
        self.tau = 1.5
        self.adjust_interval = 25
        self.z_prev = self.z.copy()
        
    def solve(self, max_iter=100, tol=1e-3):
        diag_P = np.diag(self.P)

        A_sq_sum = (self.A ** 2).sum(axis=0)

        for iter in range(max_iter):
            diag_K = diag_P + self.rho * A_sq_sum
            M_diag = 1.0 / diag_K

            def Kx(x):
                return self.P @ x + self.rho * (self.A.T @ (self.A @ x))
            
            b_x = -self.q + self.rho * (self.A.T @ (self.z - self.u_dual))

            self.x = preconditioned_conjugate_gradient(Kx, b_x, M_diag, tol=1e-5, max_iter=1000)

            Ax = self.A @ self.x
            Ax_plus_u = Ax + self.u_dual
            self.z = np.clip(Ax_plus_u, self.l, self.u)

            self.u_dual += (Ax - self.z)

            if iter % self.adjust_interval == 0 and iter > 0:
                r_norm = np.linalg.norm(Ax - self.z)
                s_norm = np.linalg.norm(self.rho * self.A.T @ (self.z - self.z_prev))

                if r_norm > self.mu * s_norm:
                    new_rho = self.rho * self.tau
                elif s_norm > self.mu * r_norm:
                    new_rho = self.rho / self.tau
                else:
                    new_rho = self.rho

                new_rho = np.clip(new_rho, self.rho_min, self.rho_max)
                if new_rho != self.rho:
                    self.rho = new_rho
                    print(f"new_rho = {new_rho}")
                self.z_prev = self.z.copy()
            
            primal_res = np.linalg.norm(Ax - self.z)
            dual_res = np.linalg.norm(self.rho * self.A.T @ (self.z - self.z_prev))
            if primal_res < tol and dual_res < tol:
                print(f"Converged at iteration {iter}")
                break
            
        return self.x
    
if __name__ == "__main__":
    P = load_matrix("P.mtx")
    q = load_vector("q.txt")
    A = load_matrix("A.mtx")
    l = load_vector("l.txt")
    u = load_vector("u.txt")
    
    osqp = OSQP(P, q, A, l, u)
    x = osqp.solve(max_iter=10000, tol=1e-5)
    print("Optimal x:", x)