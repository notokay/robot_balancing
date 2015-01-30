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

inputa1 = open('equils_a1_plot_murray_params.pkl','rb')
inputa2 = open('equils_a2_plot_murray_params.pkl','rb')
inputt = open('equils_torques_plot_murray_params.pkl','rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)
trim = pickle.load(inputt)

inputa1.close()
inputa2.close()
inputt.close()

a1_useful = []
a2_useful = []
trim_useful = []

for angle_1, angle_2, t in zip(a1, a2, trim):
  if(angle_1 < 1.8 and angle_1 > -1.8):
    a1_useful.append(angle_1)
    a2_useful.append(angle_2)
    trim_useful.append(t)

outputa1 = open('equils_a1_useful_murray_params.pkl', 'wb')
outputa2 = open('equils_a2_useful_murray_params.pkl', 'wb')
outputt = open('equils_torques_useful_murray_params.pkl', 'wb')

pickle.dump(a1_useful, outputa1)
pickle.dump(a2_useful, outputa2)
pickle.dump(trim_useful, outputt)

outputa1.close()
outputa2.close()
outputt.close()
