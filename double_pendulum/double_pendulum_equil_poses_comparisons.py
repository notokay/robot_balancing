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

inputa1 = open('equils_a1_plot_murray_params_7v8.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params_7v8.pkl','rb')
a1_8v7 = pickle.load(inputa1)
a2_8v7 = pickle.load(inputa2)
inputa1.close()
inputa2.close()
inputa1 = open('equils_a1_plot_murray_params_7v8.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params_7v8.pkl','rb')
a1_9v6 = pickle.load(inputa1)
a2_9v6 = pickle.load(inputa2)
inputa1.close()
inputa2.close()
inputa1 = open('equils_a1_plot_murray_params_5v10.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params_5v10.pkl','rb')
a1_10v5 = pickle.load(inputa1)
a2_10v5 = pickle.load(inputa2)
inputa1.close()
inputa2.close()
inputa1 = open('equils_a1_plot_murray_params_4v11.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params_4v11.pkl','rb')
a1_11v4 = pickle.load(inputa1)
a2_11v4 = pickle.load(inputa2)
inputa1.close()
inputa2.close()
inputa1 = open('equils_a1_plot_murray_params_3v12.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params_3v12.pkl','rb')
a1_12v3 = pickle.load(inputa1)
a2_12v3 = pickle.load(inputa2)
inputa1.close()
inputa2.close()

def reorder(angles):
  mid_a1 = []
  mid_a2 = []
  left_a1 = []
  left_a2 = []
  right_a1 = []
  right_a2 = []
  
  for angle_1, angle_2 in angles:
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
   
  return [left_x, left_y, mid_x, mid_y, right_x, right_y] 

fig = plt.figure(figsize = (13,13))
plt.grid(b = True, which = 'both')

components = reorder(zip(a1_8v7, a2_8v7))
plt.plot(components[2], components[3], label='8:7')

components = reorder(zip(a1_9v6, a2_9v6))
plt.plot(components[2], components[3], label = '9:6')

components = reorder(zip(a1_10v5, a2_10v5))
plt.plot(components[2], components[3], label = '10:5')

components = reorder(zip(a1_11v4, a2_11v4))
plt.plot(components[2], components[3], label = '11:4')

components = reorder(zip(a1_12v3, a2_12v3))
plt.plot(components[2], components[3], label = '12:3')

plt.legend(loc='upper right')

plt.xlabel('angle_1')
plt.ylabel('angle_2')
plt.show()
