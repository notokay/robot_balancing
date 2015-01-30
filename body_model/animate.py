from sympy import symbols, simplify, lambdify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye, asarray
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
import scipy
import pickle
import math
import sympy as sp


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0.0,0.0, 0.0, 0.])
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 4.0

t = linspace(0.0, final_time, final_time * frames_per_sec)
y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

#Set up simulation


LH_x = -1*numerical_constants[0]*sin(y[:,0])
la_x = -1*numerical_constants[12]*sin(y[:,0])
LH_y = numerical_constants[0]*cos(y[:,0])

C_x = LH_x + (numerical_constants[4]/2)*cos(y[:,1] + y[:,0])
C_y = LH_y + (numerical_constants[4]/2)*sin(y[:,1] + y[:,0])

W_x = C_x + numerical_constants[8]*cos(y[:,1] + y[:,0] + 1.57)
W_y = C_y + numerical_constants[8]*sin(y[:,1] + y[:,0] + 1.57)
c_x = C_x + numerical_constants[5]*cos(y[:,1] + y[:,0] + 1.57)

B_x = W_x + numerical_constants[9]*2*cos(y[:,1] + y[:,0] + 1.57 + y[:,3])
B_y = W_y + numerical_constants[9]*2*sin(y[:,1] + y[:,0] + 1.57 + y[:,3])
b_x = W_x + numerical_constants[9]*cos(y[:,1] + y[:,0] + 1.57 + y[:,3])

RH_x = LH_x + numerical_constants[4]*cos(y[:,1] + y[:,0])
RH_y = LH_y + numerical_constants[4]*sin(y[:,1] + y[:,0])

RA_x = RH_x + numerical_constants[12]*2*sin(y[:,2] + y[:,1] + y[:,0])
ra_x = RH_x + numerical_constants[12]*sin(y[:,2] + y[:,1] + y[:,0])
RA_y = RH_y + -1*numerical_constants[12]*2*cos(y[:,2] + y[:,1] + y[:,0])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,aspect='equal', xlim = (-2, 2), ylim = (-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time=%.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
  line.set_data([],[])
  time_text.set_text('')
  return line, time_text

def animate(i):
  thisx = [0, LH_x[i], C_x[i], W_x[i], B_x[i], W_x[i], C_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LH_y[i], C_y[i], W_y[i], B_y[i], W_y[i], C_y[i], RH_y[i], RA_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template % (i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=dt*1000, blit=True, init_func=init)
plt.show()
