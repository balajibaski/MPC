import polytope as pc
import numpy as np

def next_polytope(poly_j, poly_kappa_f, a_closed_loop):
    Hj, bj = poly_j.A, poly_j.b
    Hkf, bkf = poly_kappa_f.A, poly_kappa_f.b
    Hnext = np.vstack((Hkf, Hj @ a_closed_loop))
    bnext = np.concatenate((bkf, bj))
    return pc.Polytope(Hnext, bnext)

def determine_maximal_invariant_set(poly_kappa_f, a_closed_loop, max_iters=100):
    inv_prev = poly_kappa_f
    for _ in range(max_iters):
        inv_next = next_polytope(inv_prev, poly_kappa_f, a_closed_loop)
        if inv_prev <= inv_next:
            return inv_next
        inv_prev = inv_next
    raise Exception("Failed to compute Maximal Invariant Set")
