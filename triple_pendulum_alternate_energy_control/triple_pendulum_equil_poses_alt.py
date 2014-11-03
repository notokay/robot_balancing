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
from mpl_toolkits.mplot3d import Axes3D
from math import fabs
import matplotlib.animation as animation
from matplotlib import cm
from triple_pendulum_setup_alt import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
import pickle

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

inputa1 = open('triple_pendulum_angle_one_zoom_half.pkl','rb')
inputa2 = open('triple_pendulum_angle_two_zoom_half.pkl','rb')
inputa3 = open('triple_pendulum_angle_three_zoom_half.pkl','rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)
a3 = pickle.load(inputa3)

inputa1.close()
inputa2.close()
inputa3.close()


LH_x = []
LH_y = []
RH_x = []
RH_y = []
RA_x = []
RA_y = []
angle_1 = []
angle_2 = []
angle_3 = []
left_leg = numerical_constants[0]
hip = numerical_constants[4]
right_leg = numerical_constants[0]
"""
for i in range(len(a1)):
  if(i%10 ==0):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
a1 = angle_1
a2 = angle_2
a3 = angle_3

angle_1 =[]
angle_2 = []
angle_3 = []
"""
for i in range(len(a1)):
  if(a2[i] > -0.79  and a2[i] < 1.57):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])

a1 = angle_1
a2 = angle_2
a3 = angle_3

angle_1 =[]
angle_2 = []
angle_3 = []

for i in range(len(a1)):
  if(a3[i] > -0.79 and a3[i] < 1.57):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
a1 = angle_1
a2 = angle_2
a3 = angle_3

for i in range(len(a1)):
  LH_x.append(-1*left_leg*sin(a1[i]))
  LH_y.append(left_leg*cos(a1[i]))
  RH_x.append(-1*left_leg*sin(a1[i]) + hip*cos(a2[i] + a1[i]))
  RH_y.append(left_leg*cos(a1[i]) + hip*sin(a2[i] + a1[i]))
  RA_x.append(-1*left_leg*sin(a1[i]) + hip*cos(a2[i] + a1[i]) + right_leg*sin(a1[i]+a2[i]+a3[i]))
  RA_y.append(left_leg*cos(a1[i]) + hip*sin(a2[i] + a1[i]) + -1*right_leg*cos(a1[i] + a2[i] + a3[i]))

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, aspect = 'equal',  xlim = (-2,2), ylim = (-0.5,2))

for i in range(len(LH_x)):
  thisx = [0,LH_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LH_y[i], RH_y[i], RA_y[i]]
  plt.plot(thisx,thisy)

plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = a1
ax.scatter(a1, a2, a3, c=c)
ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')
plt.show()

outputa1 = open('triple_pen_angle_one_useful.pkl', 'wb')
outputa2 = open('triple_pen_angle_two_useful.pkl','wb')
outputa3 = open('triple_pen_angle_three_useful.pkl','wb')

pickle.dump(a1, outputa1)
pickle.dump(a2, outputa2)
pickle.dump(a3, outputa3)

outputa1.close()
outputa2.close()
outputa3.close()

