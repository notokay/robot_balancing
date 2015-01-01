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
from double_block_setup import theta1, theta2,  omega1, omega2,  l_ankle_torque, l_hip_torque,  coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
from utils import det_controllable
init_vprinting()
import pickle

inputx = open('double_block_angle_one.pkl', 'rb')
inputy = open('double_block_angle_two.pkl', 'rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)

inputx.close()
inputy.close()

answer_vector = []
tor_dict = dict(zip([l_ankle_torque], [0]))
a1 = []
a2 = []
for x, y in zip(X,Y):
  answer_vector.append([x,y])
  a1.append(x)
  a2.append(y)
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

outputA = open('double_block_linearized_A_full.pkl', 'wb')
outputB = open('double_block_linearized_B_full.pkl','wb')
outputa1 = open('double_block_angle_1_zoom.pkl','wb')
outputa2 = open('double_block_angle_2_zoom.pkl','wb')

pickle.dump(A, outputA)
pickle.dump(B, outputB)
pickle.dump(a1, outputa1)
pickle.dump(a2, outputa2)

outputA.close()
outputB.close()
outputa1.close()
outputa2.close()

