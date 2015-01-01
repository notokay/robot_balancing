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
from double_pendulum_setup import speeds, coordinates, parameter_dict, forcing_vector, specified, mass_matrix, ankle_torque
import pickle

inputx = open('double_pen_angle_1.pkl', 'rb')
inputy = open('double_pen_angle_2.pkl', 'rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)
inputx.close()
inputy.close()

answer_vector = []
tor_dict = dict(zip([ankle_torque], [0]))

for x,y in zip(X,Y):
  answer_vector.append([x,y])

#Linearization
equilibrium_points = []
for element in answer_vector:
  equilibrium_points.append(concatenate((zeros(len(speeds)), element), axis=1)) 

equilibrium_dict = []

for element in equilibrium_points:
  equilibrium_dict.append(dict(zip(speeds + coordinates, element)))

#Jacobian of forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.subs(tor_dict).jacobian(specified)

#substitute in values fo rth evariables int he forcing vector
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(tor_dict)
F_B = F_B.subs(parameter_dict)
M = mass_matrix.subs(parameter_dict)

forcing_a = []
forcing_b = []
forcing_b2 = []
M = []
for element in equilibrium_dict:
  forcing_a.append(F_A.subs(element))
  forcing_b.append(F_B.subs(element))
  forcing_b2.append(F_B.subs(element)[:,1])
  M.append(mass_matrix.subs(element))

for i in range(len(M)):
  M[i] = M[i].subs(parameter_dict)
  M[i] = array(M[i].tolist(), dtype = float)
  forcing_b2[i] = array(forcing_b2[i].tolist(), dtype = float)
  forcing_b[i] = array(forcing_b[i].tolist(), dtype = float)
  forcing_a[i] = array(forcing_a[i].tolist(), dtype = float)

#state A and input B values for linearized function

A = []
B = []
B2 = []

for m, fa in zip(M, forcing_a):
  A.append( dot(inv(m),fa) )

for m, fb, fb2 in zip(M, forcing_b, forcing_b2):
  B.append(dot(inv(m), fb))
  B2.append(dot(inv(m), fb2))

outputA = open('double_pen_linearized_A_zoom.pkl','wb')
outputB = open('double_pen_linearized_B_zoom.pkl','wb')
outputB2 = open('double_pen_linearized_B2_zoom.pkl','wb')

pickle.dump(A, outputA)
pickle.dump(B, outputB)
pickle.dump(B2, outputB2)

outputA.close()
outputB.close()
outputB2.close()
