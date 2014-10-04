from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from pydy.codegen.code import generate_ode_function
from math import fabs
import matplotlib.animation as animation
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
import pickle

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

inputa1 = open('double_pen_angle_1_zoom.pkl','rb')
inputa2 = open('double_pen_angle_2_zoom.pkl','rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)

inputa1.close()
inputa2.close()

A_x = []
A_y = []
W_x = []
W_y = []
angle_1 = []
angle_2 = []
leg = numerical_constants[0]
body = 2*numerical_constants[4]

for i in range(len(a1)):
#  if(fabs(a2[i]) < 1.57):
#    angle_1.append(a1[i])
#    angle_2.append(a2[i])
  A_x.append(leg*sin(a1[i]))
  A_y.append(leg*cos(a1[i]))
  W_x.append(leg*sin(a1[i]) + body*sin(a2[i]))
  W_y.append(leg*cos(a1[i]) + body*cos(a2[i]))

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, aspect = 'equal',  xlim = (-2,2), ylim = (-0.5,2))

for i in range(len(A_x)):
  thisx = [0,A_x[i], W_x[i]]
  thisy = [0, A_y[i], W_y[i]]
  plt.plot(thisx,thisy)

plt.show()

fig = plt.figure()
plt.plot(angle_1, angle_2)
plt.show()
