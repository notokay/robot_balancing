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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, constants
import pickle

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

numerical_constants = array([0.5,  # leg_length[m]
                             7.0, # leg_mass[kg]
                             0.75, # body_length[m]
                             8.0,  # body_mass[kg]
                             9.81],    # acceleration due to gravity [m/s^2]
                             )

parameter_dict = dict(zip(constants, numerical_constants))

inputa1 = open('equils_a1_useful_murray_params.pkl','rb')
inputa2 = open('equils_a2_useful_murray_params.pkl','rb')

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
body = numerical_constants[2]

for i in range(len(a1)):
#  if(fabs(a2[i]) < 1.57):
#    angle_1.append(a1[i])
#    angle_2.append(a2[i])
  A_x.append(-1*leg*sin(a1[i]))
  A_y.append(leg*cos(a1[i]))
  W_x.append(-1*leg*sin(a1[i]) + -1*body*sin(a1[i] + a2[i]))
  W_y.append(leg*cos(a1[i]) + body*cos(a1[i] + a2[i]))

#fig = plt.figure()
#ax = fig.add_subplot(111, autoscale_on = False, aspect = 'equal',  xlim = (-2,2), ylim = (-0.0,2))

#for i in range(len(A_x)):
#  thisx = [0,A_x[i], W_x[i]]
#  thisy = [0, A_y[i], W_y[i]]
#  plt.plot(thisx,thisy)

#plt.show()

mid_a1 = []
mid_a2 = []
left_a1 = []
left_a2 = []
right_a1 = []
right_a2 = []

for angle_1, angle_2 in zip(a1, a2):
  if(angle_1 < 1.8 and angle_1 > -1.8):
    mid_a1.append(angle_1)
    mid_a2.append(angle_2)
  elif(angle_1 < -1.8 and angle_1 > -3.14):
    left_a1.append(angle_1)
    left_a2.append(angle_2)
  elif(angle_1 < 3.14 and angle_1 > 1.8):
    right_a1.append(angle_1)
    right_a2.append(angle_2)

mid_order = np.argsort(mid_a2)
mid_x = np.array(mid_a1)[mid_order]
mid_y = np.array(mid_a2)[mid_order]

left_order = np.argsort(left_a2)
left_x = np.array(left_a1)[left_order]
left_y = np.array(left_a2)[left_order]

right_order = np.argsort(right_a2)
right_x = np.array(right_a1)[right_order]
right_y = np.array(right_a2)[right_order]

fig = plt.figure(figsize = (13,13))
plt.grid(b = True, which = 'both')
plt.plot(left_x, left_y)
plt.plot(mid_x, mid_y)
plt.plot(right_x, right_y)
plt.xlabel('angle_1')
plt.ylabel('angle_2')
plt.show()
