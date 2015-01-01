# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
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
#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

#Initial Conditions for speeds and positions
x0 = zeros(4)
x0[:2] = deg2rad(40.0)
x0[1] = deg2rad(120)
#Specifies numerical constants for inertial/mass properties
numerical_constants = array([1.035,  # leg_length[m]
                             0.58,   # leg_com_length[m]
                             23.779, # leg_mass[kg]
                             0.383,  # leg_inertia [kg*m^2]
                             0.305,  # body_com_length [m]
                             32.44,  # body_mass[kg]
                             1.485,  # body_inertia [kg*m^2]
                             9.81],    # acceleration due to gravity [m/s^2]
                             )
#Set input torques to 0
numerical_specified = zeros(2)

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

#Create dictionaries for the values for the equilibrium point of (0,0) i.e. pointing straight up
equilibrium_point = zeros(len(coordinates + speeds))
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))
parameter_dict = dict(zip(constants, numerical_constants))

#Jacobian of the forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.jacobian(specified)

#Substitute in the values for the variables in the forcing vector
F_A = simplify(F_A.subs(equilibrium_dict))
F_A = F_A.subs(parameter_dict)
F_B = simplify(F_B.subs(equilibrium_dict).subs(parameter_dict))

#Convert into a floating point numpy array
F_A = array(F_A.tolist(), dtype=float)
F_B = array(F_B.tolist(), dtype=float)

M = mass_matrix.subs(equilibrium_dict)
simplify(M)

M = M.subs(parameter_dict)
M = array(M.tolist(), dtype = float)

#Compute the state A and input B values for our linearized function
A = dot(inv(M), F_A)
B = dot(inv(M), F_B)

#Makes sure our function is controllable
#assert controllable(A,B)

Q = eye(4)
R = eye(2)

S = solve_continuous_are(A, B, Q, R)
K = dot(dot(inv(R), B.T), S)

torque_vector = []
time_vector = []

def controller(x,t):
  torque_vector.append([500*sin(t), -500*sin(t)])
  time_vector.append(t)
  return [500*sin(t),-500*sin(t) ]

#args['specified'] = controller

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

threshold = 0.001

while (angle_one < 3.14):
  angle_two = -3.14
  while (angle_two < 3.14):
    lam_sol = lam_f(angle_one, angle_two)
    if(lam_sol < threshold and lam_sol > -1*threshold):
      answer_vector.append([lam_sol, angle_one, angle_two, lam_w(angle_one, angle_two)])
      X.append(angle_one)
      Y.append(angle_two)
      torvec.append(lam_w(angle_one, angle_two))
#      print([lam_sol, angle_one, angle_two, lam_w(angle_one, angle_two)])
    angle_two = angle_two+0.001
  angle_one = angle_one + 0.0001
  print(angle_one)

plt.scatter(X,Y)
xlabel('angle_1')
ylabel('angle_2')

plt.show()
