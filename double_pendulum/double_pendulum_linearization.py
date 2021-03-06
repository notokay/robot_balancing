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
from double_pendulum_setup import speeds, coordinates, constants, forcing_vector, specified, mass_matrix, ankle_torque
import pickle

numerical_constants = array([0.5,  # leg_length[m]
                             7.0, # leg_mass[kg]
                             0.75, # body_length[m]
                             8.0,  # body_mass[kg]
                             9.81],    # acceleration due to gravity [m/s^2]
                             )
parameter_dict = dict(zip(constants, numerical_constants))

inputa1 = open('equils_a1_useful_murray_params.pkl', 'rb')
inputa2 = open('equils_a2_useful_murray_params.pkl', 'rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)
inputa1.close()
inputa2.close()

answer_vector = []
tor_dict = dict(zip([ankle_torque], [0]))

for angle_1, angle_2 in zip(a1,a2):
  answer_vector.append([angle_1, angle_2])
print("Answer Vector Done")

#Linearization
equilibrium_points = []
for element in answer_vector:
  equilibrium_points.append(concatenate((zeros(len(speeds)), element), axis=1)) 
print("Equilibrium Ponts done")
equilibrium_dict = []

for element in equilibrium_points:
  equilibrium_dict.append(dict(zip(speeds + coordinates, element)))
print("Equilibrium dict done")

#Jacobian of forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.subs(tor_dict).jacobian(specified)
print("Jacobian done")

#substitute in values for the evariables in the forcing vector
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(parameter_dict)
mass_matrix = mass_matrix.subs(parameter_dict)
print("Subs done")

forcing_a = []
forcing_b = []
M = []

for element in equilibrium_dict:
  forcing_a.append(F_A.subs(element))
  forcing_b.append(F_B.subs(element))
  M.append(mass_matrix.subs(element))
print("Equilibrium Done")

for i in range(len(M)):
  M[i] = array(M[i].tolist(), dtype = float)
  forcing_b[i] = array(forcing_b[i].tolist(), dtype = float)
  forcing_a[i] = array(forcing_a[i].tolist(), dtype = float)

#state A and input B values for linearized function

A = []
B = []

for m, fa in zip(M, forcing_a):
  A.append(dot(inv(m),fa))

for m, fb in zip(M, forcing_b):
  B.append(dot(inv(m), fb))

outputA = open('double_pendulum_linearized_A_murray_params.pkl','wb')
outputB = open('double_pendulum_linearized_B_murray_params.pkl','wb')

pickle.dump(A, outputA)
pickle.dump(B, outputB)

outputA.close()
outputB.close()

