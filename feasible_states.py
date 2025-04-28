import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import polytope as pc
from polytope import extreme
from scipy.spatial import ConvexHull

# --- Helper function to plot polytopes ---
def plot_polytope(poly, ax, facecolor='none', edgecolor='k', linestyle='-', linewidth=2, label=None):
    """Plot a 2D polytope with filled color."""
    try:
        vertices = extreme(poly)
    except Exception as e:
        print("Warning: Could not compute extreme points:", e)
        return

    if vertices is None or len(vertices) == 0:
        print("Warning: Polytope has no vertices.")
        return
    
    # Sort vertices to plot nicely
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]

    # Close the polygon
    vertices = np.vstack((vertices, vertices[0]))

    ax.plot(vertices[:, 0], vertices[:, 1],
            color=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label)
    
    ax.fill(vertices[:, 0], vertices[:, 1],
            facecolor=facecolor,
            edgecolor='none',
            alpha=0.3)  # transparent fill

# --- Main Function to Compute MPC Feasible Set ---
def compute_mpc_feasible_set(A, B, H_x, H_u, b_x, b_u, O_inf, N):
    """
    Computes the MPC feasible set X_N using backward recursion and plots it along with the terminal invariant set O_inf.
    
    Args:
        O_inf: Polytope representing the terminal invariant set (O_inf)
        N: The prediction horizon length (N)
    """
    # Initialize for backward recursion
    H_infty = O_inf.A
    b_infty = O_inf.b

    X_N = O_inf  # Start from O_inf

    # Backward recursion
    for i in range(N):
        print(f"Backward step {i+1}")

        # Build (x,u) constraint
        H1_last_row_block = np.hstack((H_infty @ A, H_infty @ B))
        H_z = np.vstack((spla.block_diag(H_x, H_u), H1_last_row_block))
        b_z = np.vstack((
            b_x.reshape(-1, 1),
            b_u.reshape(-1, 1),
            b_infty.reshape(-1, 1)
        ))

        # Define (x,u) polytope
        S = pc.Polytope(H_z, b_z)

        # Project onto x only
        n_x = A.shape[0]
        x_indices = list(range(1, n_x+1))  # because polytope expects 1-based indices
        X_N = S.project(x_indices)

        # Update for next iteration
        H_infty = X_N.A
        b_infty = X_N.b

    return X_N



