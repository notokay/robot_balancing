from sympy import symbols, simplify, lambdify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye
from numpy.linalg import inv
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.physics.vector import init_vprinting, vlatex
from math import fabs
init_vprinting()
import pickle
from mpl_toolkits.mplot3d import Axes3D


# Control
# =======

inputK = open('double_pendulum_LQR_K_murray_params.pkl','rb')
inputa1 = open('equils_a1_useful_murray_params.pkl','rb')
inputa2 = open('equils_a2_useful_murray_params.pkl','rb')

K = pickle.load(inputK)
a1 = pickle.load(inputa1)
a1 = np.asarray(a1, dtype = float)
a2 = pickle.load(inputa2)
a2 = np.asarray(a2, dtype = float)

inputK.close()
inputa1.close()
inputa2.close()

theta1_theta1 = []
theta1_theta2 = []
theta1_omega1 = []
theta1_omega2 = []

theta2_theta1 = []
theta2_theta2 = []
theta2_omega1 = []
theta2_omega2 = []

for element in K:
  theta2_theta1.append(element[1][0])
  theta2_theta2.append(element[1][1])
  theta2_omega1.append(element[1][2])
  theta2_omega2.append(element[1][3])

A = []
b1 = []

for t1, t2  in zip(a1, a2):
  A.append([1, t1, t1**2, t1**3, t2, t2**2, t2**3, 0.000001, 0.000001**2, 0.000001**3, 0.000001, 0.000001**2, 0.000001**3])

t2t1_eq = np.linalg.lstsq(A, theta2_theta1)[0]
t2t2_eq = np.linalg.lstsq(A, theta2_theta2)[0]
t2o1_eq = np.linalg.lstsq(A, theta2_omega1)[0]
t2o2_eq = np.linalg.lstsq(A, theta2_omega2)[0]

gain_coefs = array([t2t1_eq, t2t2_eq, t2o1_eq, t2o2_eq])

outputG = open('double_pendulum_gain_coefs_murray_params.pkl', 'wb')
pickle.dump(gain_coefs, outputG)

outputG.close()
