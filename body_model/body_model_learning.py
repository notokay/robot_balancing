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
from sklearn import svr
import pickle

inputx = open('bm_success_x.pkl', 'rb')
inputtor = open('bm_success_tor.pkl','rb')

x_vec = pickle.load(inputx)
torque_vector = pickle.load(inputtor)

inputx.close()
inputtor.close()

x_vec = x_vec[:4000]
torque_vector = torque_vector[:4000]

svc = svm.SVC(kernel='RBF')

hip_tor = []

for element in torque_vector:
  hip_tor.append(element[1])

svc.fit(x_vec, hip_tor)
