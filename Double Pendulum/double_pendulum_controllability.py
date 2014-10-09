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
#from utils import controllable

init_vprinting()

inputx = open('double_pen_angle_1.pkl', 'rb')
inputy = open('double_pen_angle_2.pkl', 'rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)

inputx.close()
inputy.close()

answer_vector = []

for x, y in zip(X,Y):
  answer_vector.append([x,y])

#Linearization
equilibrium_points = []
for element in answer_vector:
  equilibrium_points.append(concatenate((element, zeros(len(speeds))), axis=1)) 

equilibrium_dict = []

for element in equilibrium_points:
  equilibrium_dict.append(dict(zip(coordinates + speeds, element)))

#Jacobian of forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.jacobian(specified)

#substitute in values fo rth evariables int he forcing vector
F_A = simplify(F_A.subs(parameter_dict))
F_B = simplify(F_B.subs(parameter_dict))

forcing_a = []
forcing_b = []
M = []
for element in equilibrium_dict:
  forcing_a.append(F_A.subs(element))
  forcing_b.append(F_B.subs(element)[:,1])
  M.append(mass_matrix.subs(element))

for i in range(len(M)):
  M[i] = M[i].subs(parameter_dict)
  M[i] = array(M[i].tolist(), dtype = float)
  forcing_b[i] = array(forcing_b[i].tolist(), dtype = float)
  forcing_a[i] = array(forcing_a[i].tolist(), dtype = float)

#state A and input B values for linearized function

A = []
B = []
controllability_det = []

for m, fa in zip(M, forcing_a):
  A.append(dot(inv(m),fa) )

for m, fb in zip(M, forcing_b):
  B.append(dot(inv(m), fb))

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
