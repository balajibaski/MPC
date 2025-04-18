import numpy as np
import polytope as pc
import control as ctrl

# System dynamics
A = np.array([
    [0.9835, 2.782, 0],
    [-0.0006821, 0.978, 0],
    [-0.000973, 2.804, 1]
])

B = np.array([
    [0.01293],
    [0.00100],
    [0.001425]
])

# Constraints
alpha_max = 11.5 * np.pi / 180
q_max = 14 * np.pi / 180
theta_max = 35 * np.pi / 180
gamma_max = 23 * np.pi / 180
x_max = np.array([[alpha_max], [q_max], [theta_max], [gamma_max]])
x_min = -x_max
delta_min = -24 * np.pi / 180
delta_max = 27 * np.pi / 180

# Hx, bx
H_state = np.vstack((np.eye(3), -np.eye(3)))
b_state = np.vstack((x_max[:3], -x_min[:3]))
H_gamma = np.array([[1, 0, -1], [-1, 0, 1]])
b_gamma = np.array([[gamma_max], [gamma_max]])
H_x = np.vstack((H_state, H_gamma))
b_x = np.vstack((b_state, b_gamma)).flatten()

# Hu, bu
H_u = np.array([[1], [-1]])
b_u = np.array([[delta_max], [-delta_min]]).flatten()

# Create polytopes
X = pc.Polytope(H_x, b_x)
U = pc.Polytope(H_u, b_u)

# LQR controller
Q = np.eye(3)
R = np.array([[1]])
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
Acl = A + B @ K

# Terminal set
H_kappa_f = np.vstack((H_x, H_u @ K))
b_kappa_f = np.vstack((b_x.reshape(-1, 1), b_u.reshape(-1, 1)))
X_kappa_f = pc.Polytope(H_kappa_f, b_kappa_f)
