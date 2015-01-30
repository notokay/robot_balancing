# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi, concatenate, dot
from mpl_toolkits.mplot3d import Axes3D
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

inputa1 = open('equils_a1_plot_murray_params.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params.pkl', 'rb')
inputc = open('double_pendulum_controllability_plot_murray_params.pkl', 'rb')
inputt = open('equils_torques_plot_murray_params.pkl', 'rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)
c = pickle.load(inputc)
t = pickle.load(inputt)

inputa2.close()
inputa1.close()
inputc.close()
inputt.close()

c_plot = []
t_plot = []

for angle_1, angle_2, contr, trim in zip(a1, a2, c, t):
    if(angle_1 < 1.8 and angle_1 > -1.8 and angle_2 < 2.0 and angle_2 > -2.0 ):
        c_plot.append(contr)
        t_plot.append(trim)


fig = plt.figure()
plt.scatter(t_plot, c_plot)

plt.xlabel('trim')
plt.ylabel('controllability')
plt.show()
