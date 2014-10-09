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
import pickle
from double_block_setup import theta1, theta2, omega1, omega2, l_ankle_torque, l_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants

init_vprinting()

#Create dictionaries for the values for the values
zero_speed_dict = dict(zip(speeds, zeros(len(speeds))))
torque_dict = dict(zip([l_ankle_torque], [0]))

forcing_matrix = kane.forcing

forcing_matrix = simplify(forcing_matrix)

forcing_matrix = simplify(forcing_matrix.subs(zero_speed_dict).subs(parameter_dict).subs(torque_dict))

forcing_solved = solve(forcing_matrix, [l_hip_torque, sin(theta1)])

lam_l = lambdify((theta1, theta2), forcing_solved[l_hip_torque])

lam_f = lambdify((theta1, theta2), forcing_matrix[0])

x = -3.15
y = -3.15
X = []
Y = []

answer_vector = []
trim = []
threshold = 0.01

while x < 3.15:
  y = -3.15
  while y < 3.15:
    lam_sol = lam_f(x,y)
    if(lam_sol < threshold and lam_sol > -1*threshold):
      answer_vector.append([lam_sol, x, y])
      X.append(x)
      Y.append(y)
      trim.append([lam_l(x,y)])
    y = y + 0.001
  print(x)
  x = x + 0.001

outputx = open('double_block_angle_one.pkl', 'wb')
outputy = open('double_block_angle_two.pkl', 'wb')
outputtor = open('double_block_trim.pkl', 'wb')

pickle.dump(X, outputx)
pickle.dump(Y, outputy)
pickle.dump(trim, outputtor)

outputx.close()
outputy.close()
outputtor.close()

