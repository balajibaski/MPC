# Aircraft Pitch Control using Model Predictive Control (MPC) #
This project implements a Model Predictive Control (MPC) system for an aircraft pitch dynamics model, enforcing both state and input constraints. It includes computation of invariant sets, feasible sets, and full closed-loop simulation starting from multiple initial conditions.

Project Structure
aircraft_model.py
Defines the discrete-time aircraft pitch dynamics, state and input constraints, and computes the LQR controller and terminal set for the MPC.

invariant_set.py
Contains functions to compute the Maximal Invariant Set (𝒪∞) used as the terminal constraint in the MPC formulation.

feasible_states.py
Implements functions to:

Compute the MPC Feasible Set (X_N) using backward recursion.

Plot 2D projections of polytopes for visualization.

main_simulation.py
The main script that:

Computes the maximal invariant set (𝒪∞).

Computes the feasible set (X_N) for a given prediction horizon (N).

Simulates the closed-loop MPC system from selected initial points.

Plots the resulting state trajectories, control inputs, and feasible sets.

Key Concepts
Aircraft Dynamics:
A 3-state system ([Angle of Attack (α), Pitch Rate (q), Pitch Angle (θ)]) controlled via elevator deflection (δ).

Constraints:
Limits on states and control inputs based on physical aircraft properties.

LQR Controller:
Used for terminal control, ensuring stability inside the terminal set.

Maximal Invariant Set (𝒪∞):
The largest set of states for which there exists a control law that keeps the system inside the set indefinitely.

MPC Feasible Set (X_N):
Set of initial states from which the system can be driven to 𝒪∞ within N steps while satisfying all constraints.

Software Requirements
1) Python 3.8+
2) numpy
3) matplotlib
4) scipy
5) qpsolvers
6) polytope
7) control

In additional files folder, you can find the different methods of controler tried and the best suited method is executed and discussed. 
