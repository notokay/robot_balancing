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

inputA = open('triple_pen_linearized_A.pkl', 'rb')
inputB = open('triple_pen_linearized_B.pkl','rb')
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
for a,b,b2,b3 in zip(A,B,B_two, B_three):
  controllability_det.append(controllable(a,b))
  controllability_det_two.append(det_controllable(a,b2))
  controllability_det_three.append(det_controllable(a,b3))

outputdet = open('triple_pen_controllability_bool.pkl','wb')
outputdet2 = open('triple_pen_controllability_angle_two_lots.pkl', 'wb')
outputdet3 = open('triple_pen_controllability_angle_three_lots.pkl','wb')

pickle.dump(controllability_det, outputdet)
pickle.dump(controllability_det_two, outputdet2)
pickle.dump(controllability_det_three, outputdet3)

outputdet.close()
outputdet2.close()
outputdet3.close()  
