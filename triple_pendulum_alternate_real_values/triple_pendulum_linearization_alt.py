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
from triple_pendulum_setup_alt import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
init_vprinting()
import pickle

inputx = open('triple_pen_angle_one_useful.pkl', 'rb')
inputy = open('triple_pen_angle_two_useful.pkl', 'rb')
inputz = open('triple_pen_angle_three_useful.pkl', 'rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)
Z = pickle.load(inputz)

inputx.close()
inputy.close()
inputz.close()

answer_vector = []

for x, y, z in zip(X,Y,Z):
  answer_vector.append([x,y,z])
print("answer_vector done")

#Linearization
equilibrium_points = []
for element in answer_vector:
  equilibrium_points.append(concatenate((element, zeros(len(speeds))), axis=1))
print("equilibrium_points done")
equilibrium_dict = []
for element in equilibrium_points:
  equilibrium_dict.append(dict(zip(coordinates + speeds, element)))
print("equilibrium_dict done")

#Jacobian fo forcing vector w.r.t. states and inputs
tor_dict = dict(zip([l_ankle_torque], [0]))
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.subs(tor_dict).jacobian(specified)
print("jacobian done")

#Substitute in values for the variables in the forcing vector
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(parameter_dict)
print("subs done")

forcing_a = []
forcing_b = []
M = []

#Create the vectors storing jacobians about every equilibrium point
for element in equilibrium_dict:
  forcing_a.append(F_A.subs(element))
  forcing_b.append(F_B.subs(element))
  M.append(mass_matrix.subs(element))
print("forcing done")

for i in range(len(M)):
  M[i] = M[i].subs(parameter_dict)
  M[i] = array(M[i].tolist(), dtype = float)
  forcing_b[i] = array(forcing_b[i].tolist(), dtype = float)
  forcing_a[i] = array(forcing_a[i].tolist(), dtype = float)
print("m done")

#state A and input B values for linearized functions
A = []
B = []

for m, fa in zip(M, forcing_a):
  A.append(dot(inv(m), fa))
print("fa done")

for m,fb in zip(M,forcing_b):
  B.append(dot(inv(m), fb))
print("fb done")

outputA = open('triple_pen_linearized_A_useful.pkl', 'wb')
outputB = open('triple_pen_linearized_B_useful.pkl','wb')
#outputB2 = open('triple_pen_linearized_B2.pkl', 'wb')
#outputB3 = open('triple_pen_linearized_B3.pkl', 'wb')

pickle.dump(A, outputA)
pickle.dump(B, outputB)
#pickle.dump(B_two, outputB2)
#pickle.dump(B_three, outputB3)

outputA.close()
outputB.close()
#outputB2.close()
#outputB3.close()
