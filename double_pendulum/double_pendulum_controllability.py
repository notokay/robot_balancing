# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi, concatenate, dot
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp, solve, asin, acos, lambdify
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
from utils import det_controllable
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified
import pickle

init_vprinting()

inputA = open('double_pendulum_linearized_A_murray_params_7v8.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_7v8.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_7v8.pkl', 'rb')

A = pickle.load(inputA)
B = pickle.load(inputB)
a2 = pickle.load(inputa2)

inputA.close()
inputB.close()
inputa2.close()

#state A and input B values for linearized function
controllability_det = []

for a, b in zip(A,B):
  controllability_det.append(det_controllable(a, b[:,1].reshape(4,1)))

outputC = open('double_pendulum_controllability_plot_murray_params_7v8.pkl', 'wb')
pickle.dump(controllability_det, outputC)
outputC.close()

order = np.argsort(a2)
xs = np.array(a2)[order]
ys = np.array(controllability_det)[order]

fig = plt.figure(figsize = (13,13))
plt.grid(b = True, which = 'both')
plt.plot(xs, ys)
plt.xlabel('angle_2')
plt.ylabel('determinant')

plt.show()
