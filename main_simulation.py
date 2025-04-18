import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
from qpsolvers import solve_qp

from aircraft_model import A, B, H_x, b_x, H_u, b_u, Q, R, P, K, Acl, X_kappa_f
from invariant_set import determine_maximal_invariant_set

# Compute terminal invariant set
O_inf = determine_maximal_invariant_set(X_kappa_f, Acl)
print("Computed maximal invariant set O_inf")
print("O_inf.A (H matrix):", O_inf.A)
print("O_inf.b (h vector):", O_inf.b)

# MPC setup
chosen_solver = 'cvxopt'
N = 10
nx, nu = A.shape[0], B.shape[1]
Q_bar = spla.block_diag(*([Q] * N + [P]))
R_bar = spla.block_diag(*([R] * N))
H = spla.block_diag(Q_bar, R_bar)

x_cl = [O_inf.chebXc]
u_cl = []
T_sim = 25

for t in range(T_sim):
    x0 = x_cl[-1].reshape(-1, 1)

    Ax = np.zeros(((N+1)*nx, (N+1)*nx + N*nu))
    Bu = np.zeros_like(Ax)
    for i in range(N+1):
        Ax[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = -np.eye(nx)
        if i > 0:
            Ax[i*nx:(i+1)*nx, (i-1)*nx:i*nx] += A
            Bu[i*nx:(i+1)*nx, (N+1)*nx + (i-1)*nu:(N+1)*nx + i*nu] = B
    Ax[0:nx, 0:nx] = np.eye(nx)
    Aeq = Ax + Bu
    beq = np.vstack((x0, np.zeros((N*nx, 1))))

    Hx_block = np.kron(np.eye(N), H_x)
    bx_block = np.tile(b_x, (N, 1))
    Hu_block = np.kron(np.eye(N), H_u)
    bu_block = np.tile(b_u, (N, 1))
    Hf = O_inf.A
    bf = O_inf.b.reshape(-1, 1)

    G = np.zeros((Hx_block.shape[0] + Hu_block.shape[0] + Hf.shape[0], (N+1)*nx + N*nu))
    h = np.vstack((bx_block.reshape(-1, 1), bu_block.reshape(-1, 1), bf.reshape(-1, 1)))
    G[:Hx_block.shape[0], :N*nx] = Hx_block
    G[Hx_block.shape[0]:Hx_block.shape[0]+Hu_block.shape[0], (N+1)*nx:] = Hu_block
    G[-Hf.shape[0]:, N*nx:(N+1)*nx] = Hf

    q = np.zeros(((N+1)*nx + N*nu,))
    z = solve_qp(P=H, q=q, G=G, h=h, A=Aeq, b=beq, solver=chosen_solver)

    if z is not None:
        u_opt = z[(N+1)*nx:(N+1)*nx+nu]
        u_cl.append(u_opt)
        x_next = A @ x0 + B @ u_opt.reshape(-1, 1)
        x_cl.append(x_next.flatten())
    else:
        print(f"QP could not be solved at time step {t}")
        break

x_cl = np.array(x_cl)
u_cl = np.array(u_cl)

# Plot results
plt.figure(figsize=(10, 6))
labels = ['Angle of Attack (α)', 'Pitch Rate (q)', 'Pitch Angle (θ)']
for i in range(3):
    plt.subplot(4, 1, i+1)
    plt.plot(x_cl[:, i], label=labels[i])
    plt.ylabel(labels[i].split()[0])
    plt.grid(True)
    plt.legend()

plt.subplot(4, 1, 4)
plt.step(np.arange(len(u_cl)), u_cl[:, 0], label='Elevator Deflection (δ)', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('δ (rad)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
