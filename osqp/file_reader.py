import scipy.io
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite

# Load .mat file
data = scipy.io.loadmat("CONT-050.mat")

# Extract matrices and vectors
P = data["P"]  # Quadratic term
A = data["A"]  # Constraint matrix
q = np.array(data["q"]).flatten()  # Linear term
l = np.array(data["l"]).flatten()  # Lower bounds
u = np.array(data["u"]).flatten()  # Upper bounds

# Save sparse matrices in Matrix Market format
mmwrite("P.mtx", P)
mmwrite("A.mtx", A)

# Save vectors in simple text format
np.savetxt("q.txt", q, fmt="%.8f")
np.savetxt("l.txt", l, fmt="%.8f")
np.savetxt("u.txt", u, fmt="%.8f")

print("Files saved: P.mtx, A.mtx, q.txt, l.txt, u.txt")
