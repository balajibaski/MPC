import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import time  
from qpsolvers import solve_qp
import polytope as pc
from polytope import extreme 
from scipy.spatial import ConvexHull

from aircraft_model import A, B, H_x, b_x, H_u, b_u, Q, R, P, K, Acl, X_kappa_f
from invariant_set import determine_maximal_invariant_set
from feasible_states import compute_mpc_feasible_set, plot_polytope

# Compute terminal invariant set
O_inf = determine_maximal_invariant_set(X_kappa_f, Acl)
print("Computed maximal invariant set O_inf")
print("O_inf.A (H matrix):", O_inf.A)
print("O_inf.b (h vector):", O_inf.b)

N = 5

X_N = compute_mpc_feasible_set(A, B, H_x, H_u, b_x, b_u, O_inf, N)

# Extract extreme points using ConvexHull
extreme_points = extreme(X_N)
# Randomly select 3 points
np.random.seed(40)
selected_points = extreme_points[np.random.choice(extreme_points.shape[0], 3, replace=False), :]
# MPC setup
chosen_solver = 'cvxopt'
nx, nu = A.shape[0], B.shape[1]
Q_bar = spla.block_diag(*([Q] * N + [P]))
R_bar = spla.block_diag(*([R] * N))
H = spla.block_diag(Q_bar, R_bar)

# Simulate from each vertex
all_x_cl = []
all_u_cl = []
T_sim = 20

for vertex in selected_points:
    x_cl = [vertex.reshape(-1)]  # Start from each vertex
    u_cl = []
    solve_times = []  
    for t in range(T_sim):
        x0 = x_cl[-1].reshape(-1, 1)

        Ax = np.zeros(((N+1)*nx, (N+1)*nx + N*nu))
        Bu = np.zeros_like(Ax)
        for i in range(N+1):
            Ax[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = -np.eye(nx)
            if i > 0:
                Ax[i*nx:(i+1)*nx, (i-1)*nx:i*nx] += A
        for i in range(1,N+1):        
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
        start_time = time.time()
        z = solve_qp(P=H, q=q, G=G, h=h, A=Aeq, b=beq, solver=chosen_solver)
        end_time = time.time()

        if z is not None:
            solve_times.append(end_time - start_time)
            u_opt = z[(N+1)*nx:(N+1)*nx+nu]
            u_cl.append(u_opt)
            x_next = A @ x0 + B @ u_opt.reshape(-1, 1)
            x_cl.append(x_next.flatten())
        else:
            print(f"QP could not be solved at time step {t}")
            break
    all_x_cl.append(np.array(x_cl))
    all_u_cl.append(np.array(u_cl))    

    if solve_times:
        avg_time = np.mean(solve_times)
        max_time = np.max(solve_times)
        print(f"Average QP solve time: {avg_time:.6f} seconds")
        print(f"Maximum QP solve time: {max_time:.6f} seconds")

# Plot results
plt.figure(figsize=(10, 6))
labels = ['Angle of Attack (α)', 'Pitch Rate (q)', 'Pitch Angle (θ)']

# Plot state trajectories
for traj_idx, traj in enumerate(all_x_cl):
    for i in range(3):
        plt.subplot(4, 1, i+1)
        plt.plot(traj[:, i], alpha=0.5, label=labels[i] if traj_idx == 0 else None)
        plt.ylabel(labels[i].split()[0])
        plt.grid(True)
    if traj_idx == 0:
        plt.legend()

# Plot control inputs for all trajectories
plt.subplot(4, 1, 4)
for u_traj in all_u_cl:
    plt.step(np.arange(len(u_traj)), u_traj[:, 0], alpha=0.5, linestyle='--')
plt.xlabel('Time step')
plt.ylabel('δ (rad)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final projection for 2D plotting
X_N_2d = X_N.project([1, 2])
O_inf_2d = O_inf.project([1, 2])

# Plotting
fig, ax = plt.subplots()

plot_polytope(X_N_2d, ax=ax, facecolor='lightblue', edgecolor='blue', label=r'$X_N$')
plot_polytope(O_inf_2d, ax=ax, facecolor='none', edgecolor='red', linestyle='--', label=r'$\mathcal{O}_\infty$')

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title(r"Feasible Set $X_N$ and Terminal Set $\mathcal{O}_\infty$")
ax.legend()
ax.grid(True)

plt.show()

