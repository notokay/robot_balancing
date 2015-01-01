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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants
import pickle

init_vprinting()

inputA = open('double_pen_linearized_A.pkl', 'rb')
inputB = open('double_pen_linearized_B.pkl', 'rb')

A = pickle.load(inputA)
B = pickle.load(inputB)

inputA.close()
inputB.close()

#state A and input B values for linearized function
controllability_det = []

for a,b in zip(A,B):
  controllability_det.append(det_controllable(a, b))


plt.scatter(X,Y)
plt.xlabel('angle_1')
plt.ylabel('angle_2')
plt.show()

plt.scatter(Y, controllability_det)
plt.xlabel('angle_2')
plt.ylabel('determinant')

plt.show()
