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


# Control
# =======

inputK = open('bm_LQR_K_useful.pkl','rb')
inputa1 = open('bm_angle_one_useful.pkl','rb')
inputa2 = open('bm_angle_two_useful.pkl','rb')
inputa3 = open('bm_angle_three_useful.pkl','rb')
inputa4 = open('bm_angle_four_useful.pkl','rb')

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
x = [-0.01,0.0,0,0,0,0,0,0]
tor1 = []
tor2 = []
tor3 = []
for element in K:
  output_torque = -dot(element, x)
  tor1.append(output_torque[1])

plt.scatter(a1, tor1)
xlabel('angle 1')
ylabel('torque value')
plt.show()


"""
f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

#a1 = a1.reshape(len(a1), 1)
#a2 = a2.reshape(len(a2), 1)
#a3 = a3.reshape(len(a3), 1)
#a4 = a4.reshape(len(a4), 1)

ax1.scatter(range(len(tor1)), tor1)
ax1.set_xlabel('angle 1')
ax1.set_ylabel('gain value')

ax2.scatter(a2, tor1)
ax2.set_xlabel('angle 2')
ax2.set_ylabel('gain value')

ax3.scatter(a3, tor1)
ax3.set_xlabel('angle 3')
ax3.set_ylabel('gain value')

ax4.scatter(a4, tor1)
ax4.set_xlabel('angle 4')
ax4.set_ylabel('gain value')

plt.show()
"""
