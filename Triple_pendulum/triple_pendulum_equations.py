from sympy import symbols, simplify, trigsimp, solve, latex, diff, cos, sin
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
#from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, pi, zeros, dot, eye
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.utilities import lambdify
from sympy.physics.vector import init_vprinting, vlatex
from triple_pendulum_setup import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
init_vprinting()

time = symbols('t')

#Create dictionaries for the values for the values
zero_speed_dict = dict(zip(speeds, zeros(len(speeds))))
parameter_dict = dict(zip(constants, numerical_constants))
torque_dict = dict(zip([l_ankle_torque], [0]))

forcing_matrix = kane.forcing

forcing_matrix = simplify(forcing_matrix)

forcing_matrix = simplify(forcing_matrix.subs(zero_speed_dict).subs(parameter_dict).subs(torque_dict))

forcing_solved = solve(forcing_matrix, [l_hip_torque, r_hip_torque, sin(theta1)])

lam_l = lambdify((theta1, theta2, theta3), forcing_solved[l_hip_torque])

lam_r = lambdify((theta1, theta2, theta3), forcing_solved[r_hip_torque])

lam_f = lambdify((theta1, theta2, theta3), forcing_matrix[0])

x = -3.14
y = -3.14
z = -3.14
X = []
Y = []
Z = []

answer_vector = []

threshold = 0.1

while x < 3.14:
  y = -1.57
  z = -1.57
  while y < 3.14:
    z = -1.57
    while z < 3.14:
      lam_sol = lam_f(x,y,z)
      if(lam_sol < threshold and lam_sol > -1*threshold):
        answer_vector.append([lam_sol,lam_l(x,y,z), lam_r(x,y,z), x, y, z])
        X.append(x)
        Y.append(y)
        Z.append(z)
      z = z + 0.01
    y = y + 0.01
  print(x)
  x = x + 0.01

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = X
ax.scatter(X, Y, Z, c = c)
#ax.plot_trisurf(X,Y,Z)

ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')

plt.show()
