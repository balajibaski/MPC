# Importing the Necessary Packages
from qpsolvers import solve_qp as sol
import numpy as np

'''
Write the Necessary Documentation here
'''


# System Dynamics
A = np.array([[0.9835, 2.782, 0], 
              [-0.0006821, 0.978, 0],
              [0, 0, 1]])

B = np.array([[0.01293],
              [0.00100],
              [0.001425]])