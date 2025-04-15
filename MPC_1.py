import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
from qpsolvers import solve_qp

def build_prediction_matrices(A, B, N):
    n = A.shape[0]
    A_bar = np.zeros((n * N, n))
    B_bar = np.zeros((n * N, N))

    for i in range(N):
        A_bar[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            B_bar[i*n:(i+1)*n, j] = (np.linalg.matrix_power(A, i-j) @ B).flatten()
    return A_bar, B_bar

def build_cost_matrices(A_bar, B_bar, Q, R, Qf, x0, N):
    Q_block = block_diag(*([Q] * (N-1) + [Qf]))
    R_block = R * np.eye(N)
    P = B_bar.T @ Q_block @ B_bar + R_block
    q = (A_bar @ x0).T @ Q_block @ B_bar
    return P, q

def build_constraints(A_bar, B_bar, x0, x_min, x_max, u_min, u_max, N):
    G_list = []
    h_list = []

    for i in range(N):
        Gx = B_bar[i*3:(i+1)*3, :]
        Ax_x0 = A_bar[i*3:(i+1)*3, :] @ x0
        G_list.append(Gx)
        h_list.append(x_max - Ax_x0)
        G_list.append(-Gx)
        h_list.append(-(x_min - Ax_x0))

    Gu = np.vstack([np.eye(N), -np.eye(N)])
    hu = np.hstack([np.ones(N)*u_max, -np.ones(N)*u_min])
    G_list.append(Gu)
    h_list.append(hu)

    G = np.vstack(G_list)
    h = np.hstack(h_list)
    return G, h

# System definition
A = np.array([
    [0.9835, 2.782, 0],
    [-0.0006821, 0.978, 0],
    [-0.0009730, 2.804, 1]
])
B = np.array([
    [0.01293],
    [0.00100],
    [0.001425]
])

Q = np.eye(3)
Qf = Q
R = 1.0
N = 10

x0 = np.array([0.01, 0.01, 0.01])
x_min = np.array([-1.0, -1.0, -1.0])
x_max = np.array([ 1.0,  1.0,  1.0])
u_min = -1.0
u_max =  1.0

# Build everything
A_bar, B_bar = build_prediction_matrices(A, B, N)
P, q = build_cost_matrices(A_bar, B_bar, Q, R, Qf, x0, N)
G, h = build_constraints(A_bar, B_bar, x0, x_min, x_max, u_min, u_max, N)

# Convert to sparse
P = sp.csc_matrix(P)
G = sp.csc_matrix(G)

# Solve QP
z = solve_qp(P, q, G, h, solver="osqp")

print("Optimal inputs z:\n", z)
