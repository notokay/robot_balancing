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
import pickle

inputA = open('triple_pen_linearized_A_useful.pkl','rb')
inputB = open('triple_pen_linearized_B_useful.pkl','rb')
input_one = open('triple_pendulum_angle_one_zoom_half.pkl','rb')
input_two = open('triple_pendulum_angle_two_zoom_half.pkl','rb')
input_three = open('triple_pendulum_angle_three_zoom_half.pkl','rb')

A = pickle.load(inputA)
B = pickle.load(inputB)
theta1 = pickle.load(input_one)
theta2 = pickle.load(input_two)
theta3 = pickle.load(input_three)

inputA.close()
inputB.close()
input_one.close()
input_two.close()
input_three.close()

Q = ((1/.6)**2)*eye(6)
Q[0][0] = ((1.0/0.01)**2)
Q[1][1] = ((1.0/1.0)**2)
Q[2][2] = ((1.0/1.0)**2)
Q[3][3] = ((1.0/0.01)**2)
Q[4][4] = ((1.0/0.01)**2)
Q[5][5] = ((1.0/0.01)**2)
R = eye(3)
R[0][0] = (1/0.000000000000000001)**2
R[1][1] = ((1.0/1.0)**2)
R[2][2] = ((1.0/1.0)**2)

K = []

for a,b in zip(A,B):
  S = solve_continuous_are(a,b,Q,R)
  K.append(dot(dot(inv(R), b.T), S))

outputK = open('triple_pen_LQR_K_useful.pkl', 'wb')

pickle.dump(K,outputK)

outputK.close()
