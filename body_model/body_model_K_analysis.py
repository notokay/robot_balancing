from sympy import symbols, simplify, lambdify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from body_model_setup import theta1, theta2, theta3,theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque, waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants, l_leg, r_leg, crotch, body
from sympy.physics.vector import init_vprinting, vlatex
from math import fabs
init_vprinting()
import pickle
from mpl_toolkits.mplot3d import Axes3D


# Control
# =======

inputK = open('bm_LQR_K_useful.pkl','rb')
inputa1 = open('bm_angle_one_useful_1.pkl','rb')
inputa2 = open('bm_angle_two_useful_1.pkl','rb')
inputa3 = open('bm_angle_three_useful_1.pkl','rb')
inputa4 = open('bm_angle_four_useful_1.pkl','rb')

K = pickle.load(inputK)
a1 = pickle.load(inputa1)
a1 = np.asarray(a1, dtype = float)
a2 = pickle.load(inputa2)
a2 = np.asarray(a2, dtype = float)
a3 = pickle.load(inputa3)
a3 = np.asarray(a3, dtype = float)
a4 = pickle.load(inputa4)
a4 = np.asarray(a4, dtype = float)


inputK.close()
inputa1.close()
inputa2.close()
inputa3.close()
inputa4.close()
x = [0., 0.7, 0.2, 0, 0,  0.0, 0, 0]
tor_self = []
tor1 = []
tor2 = []
tor3 = []
tor4 = []
tor5 = []
tor6 = []
tor7 = []
for element in K:
#  output_torque = -dot(element, x)
#  tor1.append(output_torque[1])
  tor_self.append(element[1][1])
  tor1.append(element[1][0])
  tor2.append(element[1][2])
  tor3.append(element[1][3])
  tor4.append(element[1][4])
  tor5.append(element[1][5])
  tor6.append(element[1][6])
  tor7.append(element[1][7])
  

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(a2, a1, tor2)
ax.set_xlabel("theta_2")
ax.set_ylabel("theta_1")
ax.set_zlabel("gain")
plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(a2, a2, tor2)
ax.set_xlabel("theta_2")
ax.set_ylabel("theta_2")
ax.set_zlabel("gain")
plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(a2, a3, tor2)
ax.set_xlabel("theta_2")
ax.set_ylabel("theta_3")
ax.set_zlabel("gain")
plt.show()
fig = plt.figure()

ax = fig.gca(projection = '3d')
ax.scatter(a1, a4, tor2)
ax.set_xlabel("theta_1")
ax.set_ylabel("theta_4")
ax.set_zlabel("gain")
plt.show()




f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

#a1 = a1.reshape(len(a1), 1)
#a2 = a2.reshape(len(a2), 1)
#a3 = a3.reshape(len(a3), 1)
#a4 = a4.reshape(len(a4), 1)

ax1.scatter(a1, tor4)
ax1.set_xlabel('angle 1')
ax1.set_ylabel('omega1 gain')

ax2.scatter(a2, tor4)
ax2.set_xlabel('angle 1')
ax2.set_ylabel('omega2 gain')

ax3.scatter(a3, tor5)
ax3.set_xlabel('angle 1')
ax3.set_ylabel('omega3 gain')

ax4.scatter(a4, tor5)
ax4.set_xlabel('angle 1')
ax4.set_ylabel('omega4 gain')

plt.show()
