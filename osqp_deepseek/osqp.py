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
            break
        beta = rsnew / rsold
        p = z + beta * p
        rsold = rsnew
    return x

class OSQP:
    def __init__(self, P, q, A, l, u):
        self.P = P  # P is a diagonal matrix (1D array or 2D array)
        self.q = q
        self.A = A
        self.l = l
        self.u = u
        self.n = q.shape[0]
        self.m = A.shape[0]
        
        # 初始化变量
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
        history = {'x': [], 'rho': []}
        
        for iter in range(max_iter):
            # x-update: 使用预处理共轭梯度法
            if len(self.P.shape) == 2:
                diag_P = np.diag(self.P)
            else:
                diag_P = self.P  # 假设P是1D的对角线元素
            
            # 计算A各列的平方和
            if isinstance(self.A, np.ndarray):
                A_sq_sum = (self.A ** 2).sum(axis=0)
            else:
                # 处理稀疏矩阵，例如scipy.sparse.csr_matrix
                A_sq_sum = np.array(self.A.power(2).sum(axis=0)).flatten()
            
            diag_K = diag_P + self.rho * A_sq_sum
            M_diag = 1.0 / diag_K
            
            # 定义Kx函数
            def Kx(x):
                return self.P @ x + self.rho * (self.A.T @ (self.A @ x))
            
            # 构建右侧项
            b_x = -self.q + self.rho * (self.A.T @ (self.z - self.u_dual))
            
            # 求解线性系统
            self.x = preconditioned_conjugate_gradient(Kx, b_x, M_diag, tol=1e-5, max_iter=1000)
            
            # z-update: 投影到[l, u]
            Ax_plus_u = self.A @ self.x + self.u_dual
            self.z = np.clip(Ax_plus_u, self.l, self.u)
            
            # u-update
            self.u_dual += (self.A @ self.x - self.z)
            
            # 动态调整rho
            if iter % self.adjust_interval == 0 and iter > 0:
                r_norm = np.linalg.norm(self.A @ self.x - self.z)
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
            
            history['x'].append(self.x.copy())
            history['rho'].append(self.rho)
            
            # 检查终止条件
            primal_res = np.linalg.norm(self.A @ self.x - self.z)
            dual_res = np.linalg.norm(self.rho * self.A.T @ (self.z - self.z_prev))
            if primal_res < tol and dual_res < tol:
                print(f"Converged at iteration {iter}")
                break
        
        return self.x, history

# 示例测试
if __name__ == "__main__":
    P = load_matrix("P.mtx")
    q = load_vector("q.txt")
    A = load_matrix("A.mtx")
    l = load_vector("l.txt")
    u = load_vector("u.txt")
    
    osqp = OSQP(P, q, A, l, u)
    x, history = osqp.solve(max_iter=10000, tol=1e-5)
    print("Optimal x:", x)