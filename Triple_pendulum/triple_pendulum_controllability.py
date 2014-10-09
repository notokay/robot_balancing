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
from utils import det_controllable, controllable
init_vprinting()
import pickle

inputA = open('triple_pen_linearized_A_useful.pkl', 'rb')
inputB = open('triple_pen_linearized_B_useful.pkl','rb')
inputB2 = open('triple_pen_linearized_B2.pkl', 'rb')
inputB3 = open('triple_pen_linearized_B3.pkl', 'rb')
A = pickle.load(inputA)
B = pickle.load(inputB)
B2 = pickle.load(inputB2)
B3 = pickle.load(inputB3)

inputA.close()
inputB.close()
inputB2.close()
inputB3.close()

controllability_det = []
controllability_det_two = []
controllability_det_three = []

#calculate determinant of controllability matrix
for a,b in zip(A,B):
  controllability_det_two.append(det_controllable(a,b[:,1].reshape(6,1)))
  controllability_det_three.append(det_controllable(a,b[:,2].reshape(6,1)))

outputdet2 = open('triple_pen_controllability_angle_two_useful.pkl', 'wb')
outputdet3 = open('triple_pen_controllability_angle_three_useful.pkl','wb')

pickle.dump(controllability_det_two, outputdet2)
pickle.dump(controllability_det_three, outputdet3)

outputdet2.close()
outputdet3.close()  
