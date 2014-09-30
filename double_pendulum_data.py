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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants
import pickle
#from utils import controllable

torque_vector = []
time_vector = []

forcing_matrix = simplify(kane.forcing)

torque_zero_dict = dict(zip([ankle_torque],[0]))

speeds_zero_dict = dict(zip([omega1, omega2], [0,0]))

subbed_eq = simplify(forcing_matrix.subs(torque_zero_dict).subs(speeds_zero_dict))

solved_eq = solve(subbed_eq, [theta1, waist_torque])

num_eq = simplify(subbed_eq.subs(parameter_dict))

sin_eq = num_eq[0]
waist_eq = -1*(num_eq[1] - waist_torque)

lam_f = lambdify((theta1, theta2),sin_eq)
lam_w = lambdify((theta1, theta2), waist_eq)

angle_one = -3.14
angle_two = -3.14
X = []
Y = []
torvec = []
answer_vector = []

threshold = 0.0001

while (angle_one < 3.14):
  angle_two = -3.14
  while (angle_two < 3.14):
    lam_sol = lam_f(angle_one, angle_two)
    if(lam_sol < threshold and lam_sol > -1*threshold):
      answer_vector.append([angle_one, angle_two])
      X.append(angle_one)
      Y.append(angle_two)
      torvec.append(lam_w(angle_one, angle_two))
    angle_two = angle_two+1
  angle_one = angle_one + 1
  print(angle_one)

#outputx = open('double_pen_angle_1.pkl', 'wb')
#outputy = open('double_pen_angle_2.pkl', 'wb')

#pickle.dump(X, outputx)
#pickle.dump(Y, outputy)

#outputx.close()
#outputy.close()
