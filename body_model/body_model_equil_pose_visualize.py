from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import fabs
import matplotlib.animation as animation
from matplotlib import cm
from body_model_setup import theta1, theta2, theta3,theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque,waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
import pickle

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

left_leg = numerical_constants[0]
hip_width = numerical_constants[4]
right_leg = numerical_constants[0]
body_height = numerical_constants[8]
body_com_height = numerical_constants[9]

a1 = 0.05
a2 = 0.0
a3 = 0.05
a4 = 0.0

LH_x = -1*left_leg*sin(a1)
LH_y = left_leg*cos(a1)
C_x = -1*left_leg*sin(a1) + (hip_width/2)*cos(a1 + a2)
C_y = left_leg*cos(a1) + (hip_width/2)*sin(a1 + a2)
W_x = -1*left_leg*sin(a1) + (hip_width/2)*cos(a1 + a2) + body_height*cos(a1 + a2 + 1.57)
W_y = left_leg*cos(a1) + (hip_width/2)*sin(a1 + a2) + body_height*sin(a1 + a2 + 1.57)
B_x = -1*left_leg*sin(a1) + (hip_width/2)*cos(a1 + a2) + body_height*cos(a1 + a2 + 1.57) + body_com_height*2*cos(a1 + a2 + 1.57 + a4)
B_y = left_leg*cos(a1) + (hip_width/2)*sin(a1 + a2) + body_height*sin(a1 + a2 + 1.57) + body_com_height*2*sin(a1 + a2 + 1.57 + a4)
RH_x = -1*left_leg*sin(a1) + hip_width*cos(a2 + a1)
RH_y = left_leg*cos(a1) + hip_width*sin(a2 + a1)
RA_x = -1*left_leg*sin(a1) + hip_width*cos(a2 + a1) + right_leg*sin(a1+a2+a3)
RA_y = left_leg*cos(a1) + hip_width*sin(a2 + a1) + -1*right_leg*cos(a1 + a2 + a3)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, aspect = 'equal',  xlim = (-2,2), ylim = (-0.5,2))

thisx = [0, LH_x, C_x, W_x, B_x,W_x, C_x, RH_x, RA_x]
thisy = [0, LH_y, C_y, W_y, B_y, W_y, C_y, RH_y, RA_y]

plt.plot(thisx,thisy)

plt.show()
