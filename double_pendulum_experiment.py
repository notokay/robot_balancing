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
#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
leg_frame = ReferenceFrame('L')
body_frame = ReferenceFrame('B')

#Sets up symbols for joint angles
theta1, theta2 = dynamicsymbols('theta1, theta2')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
body_frame.orient(leg_frame, 'Axis', (theta2, leg_frame.z))

#Sets up points for the joints and places them relative to each other
ankle = Point('A')
leg_length = symbols('l_L')
waist = Point('W')
waist.set_pos(ankle, leg_length*leg_frame.y)

#Sets up the centers of mass of each of the linkages
leg_com_length, body_com_length = symbols('d_L, d_B')
leg_mass_center = Point('L_o')
leg_mass_center.set_pos(ankle, leg_com_length*leg_frame.y)
body_mass_center = Point('B_o')
body_mass_center.set_pos(waist, body_com_length*body_frame.y)

#Sets up the angular velocities
omega1, omega2 = dynamicsymbols('omega1, omega2')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega1 - theta1.diff(),
                                    omega2 - theta2.diff()]

#Sets up the rotational axes of the angular velocities
leg_frame.set_ang_vel(inertial_frame, omega1*inertial_frame.z)
leg_frame.ang_vel_in(inertial_frame)
body_frame.set_ang_vel(leg_frame, omega2*inertial_frame.z)
body_frame.ang_vel_in(inertial_frame)

#Sets up the linear velocities of the points on the linkages
ankle.set_vel(inertial_frame, 0)
leg_mass_center.v2pt_theory(ankle, inertial_frame, leg_frame)
leg_mass_center.vel(inertial_frame)
waist.v2pt_theory(ankle, inertial_frame, leg_frame)
waist.vel(inertial_frame)
body_mass_center.v2pt_theory(waist, inertial_frame, body_frame)
body_mass_center.vel(inertial_frame)

#Sets up the masses of the linkages
leg_mass, body_mass = symbols('m_L, m_B')

#Sets up the rotational inertia of the linkages
leg_inertia, body_inertia = symbols('I_Lz, I_Bz')

#Sets up inertia dyadics
leg_inertia_dyadic = inertia(leg_frame, 0, 0, leg_inertia)
leg_central_inertia = (leg_inertia_dyadic, leg_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)
body_central_inertia = (body_inertia_dyadic, body_mass_center)

#Defines the linkages as rigid bodies
leg = RigidBody('Leg', leg_mass_center, leg_frame, leg_mass, leg_central_inertia)
body = RigidBody('Body', body_mass_center, body_frame, body_mass, body_central_inertia)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
leg_grav_force_vector = -leg_mass*g*inertial_frame.y
leg_grav_force = (leg_mass_center, leg_grav_force_vector)
body_grav_force_vector = -body_mass*g*inertial_frame.y
body_grav_force = (body_mass_center,body_grav_force_vector)

#Sets up joint torques
ankle_torque, waist_torque = dynamicsymbols('T_a, T_w')
leg_torque_vector = ankle_torque*inertial_frame.z - waist_torque*inertial_frame.z
leg_torque = (leg_frame, leg_torque_vector)

body_torque_vector = waist_torque*inertial_frame.z
body_torque = (body_frame, body_torque_vector)

#Generalized coordinates
coordinates = [theta1, theta2]

#Generalized speeds
speeds = [omega1, omega2]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [leg_grav_force,
         body_grav_force,
         leg_torque,
         body_torque]
bodies = [leg, body]

fr, frstar = kane.kanes_equations(loads, bodies)
trigsimp(fr + frstar)
mass_matrix = trigsimp(kane.mass_matrix_full)

forcing_vector = trigsimp(kane.forcing_full)

rcParams['figure.figsize'] = (14.0, 6.0)

constants = [leg_length,
             leg_com_length,
             leg_mass,
             leg_inertia,
             body_com_length,
             body_mass,
             body_inertia,
             g]
#Specified contains the matrix for the input torques
specified = [ankle_torque, waist_torque]


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
