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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified
import pickle

init_vprinting()

inputA = open('double_pendulum_linearized_A_murray_params_7v8.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_7v8.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_7v8.pkl', 'rb')
A = pickle.load(inputA)
B = pickle.load(inputB)
a2 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

inputA = open('double_pendulum_linearized_A_murray_params_6v9.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_6v9.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_6v9.pkl', 'rb')
A_6v9 = pickle.load(inputA)
B_6v9 = pickle.load(inputB)
a2_6v9 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

inputA = open('double_pendulum_linearized_A_murray_params_5v10.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_5v10.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_5v10.pkl', 'rb')
A_5v10 = pickle.load(inputA)
B_5v10 = pickle.load(inputB)
a2_5v10 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

inputA = open('double_pendulum_linearized_A_murray_params_4v11.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_4v11.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_4v11.pkl', 'rb')
A_4v11 = pickle.load(inputA)
B_4v11 = pickle.load(inputB)
a2_4v11 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

inputA = open('double_pendulum_linearized_A_murray_params_3v12.pkl', 'rb')
inputB = open('double_pendulum_linearized_B_murray_params_3v12.pkl', 'rb')
inputa2 = open('equils_a2_plot_murray_params_3v12.pkl', 'rb')
A_3v12 = pickle.load(inputA)
B_3v12 = pickle.load(inputB)
a2_3v12 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

#state A and input B values for linearized function

def c_determinant(ABa2):
  controllability_det = []
  for a,b in ABa2[0]:
    controllability_det.append(det_controllable(a, b[:,1].reshape(4,1)))
  order = np.argsort(ABa2[1])
  xs = np.array(ABa2[1])[order]
  ys = np.array(controllability_det)[order]
  return [xs, ys]

#outputC = open('double_pendulum_controllability_plot_robot_params.pkl', 'wb')
#pickle.dump(controllability_det, outputC)
#outputC.close()

fig = plt.figure(figsize = (13,13))
plt.grid(b = True, which = 'both')

xsys = c_determinant([zip(A, B), a2])
plt.plot(xsys[0], xsys[1], label = '7:8')

xsys = c_determinant([zip(A_6v9, B_6v9), a2_6v9])
plt.plot(xsys[0], xsys[1], label = '6:9')

xsys = c_determinant([zip(A_5v10, B_5v10), a2_5v10])
plt.plot(xsys[0], xsys[1], label = '5:10')

xsys = c_determinant([zip(A_4v11, B_4v11), a2_4v11])
plt.plot(xsys[0], xsys[1], label = '4:11')

xsys = c_determinant([zip(A_3v12, B_3v12), a2_3v12])
plt.plot(xsys[0], xsys[1], label = '3:12')

plt.legend(loc = 'upper right')
plt.xlabel('angle_2')
plt.ylabel('determinant')

plt.show()
