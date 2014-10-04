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

inputA = open('double_pen_linearized_A_zoom.pkl','rb')
inputB = open('double_pen_linearized_B_zoom.pkl','rb')
input_one = open('double_pen_angle_1_zoom.pkl','rb')
input_two = open('double_pen_angle_2_zoom.pkl','rb')

A = pickle.load(inputA)
B = pickle.load(inputB)
theta1 = pickle.load(input_one)
theta2 = pickle.load(input_two)

inputA.close()
inputB.close()
input_one.close()
input_two.close()

Q = ((1/0.6)**2)*eye(4)
R = eye(2)

K = []

for a,b,angle_1, angle_2 in zip(A,B,theta1, theta2):
  S = solve_continuous_are(a,b,Q,R)
  K.append(dot(dot(inv(R), b.T), S))

outputK = open('double_pen_LQR_K_zoom.pkl', 'wb')

pickle.dump(K,outputK)

outputK.close()
