from sympy import symbols, simplify, trigsimp, solve, latex, diff, cos, sin
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
#from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, pi, zeros, dot, eye
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.utilities import lambdify
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


# Orientations
# ============

theta1, theta2, theta3 = dynamicsymbols('theta1, theta2, theta3')

inertial_frame = ReferenceFrame('I')

l_leg_frame = ReferenceFrame('L')

l_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

body_frame = ReferenceFrame('B')

body_frame.orient(l_leg_frame, 'Axis', (theta2, l_leg_frame.z))

r_leg_frame = ReferenceFrame('R')

r_leg_frame.orient(body_frame, 'Axis', (theta3, body_frame.z))

# Point Locations
# ===============

# Joints
# ------

l_leg_length, hip_width = symbols('l_L, h_W')

l_ankle = Point('LA')

l_hip = Point('LH')
l_hip.set_pos(l_ankle, l_leg_length * l_leg_frame.y)

r_hip = Point('RH')
r_hip.set_pos(l_hip, hip_width * body_frame.y)

# Center of mass locations
# ------------------------

l_leg_com_length, body_com_length, r_leg_com_length, body_com_height = symbols('d_L, d_B, d_R, d_BH')

l_leg_mass_center = Point('LL_o')
l_leg_mass_center.set_pos(l_ankle, l_leg_com_length * l_leg_frame.y)

body_mass_center = Point('B_o')
body_middle = Point('B_m')
body_middle.set_pos(l_hip, body_com_length*body_frame.y)
body_mass_center.set_pos(body_middle, body_com_height*body_frame.x)

r_leg_mass_center = Point('RL_o')
r_leg_mass_center.set_pos(r_hip, -1*r_leg_com_length * r_leg_frame.y)

# Define kinematical differential equations
# =========================================

omega1, omega2, omega3 = dynamicsymbols('omega1, omega2, omega3')

time = symbols('t')

kinematical_differential_equations = [omega1 - theta1.diff(time),
                                      omega2 - theta2.diff(time),
                                      omega3 - theta3.diff(time)]

# Angular Velocities
# ==================

l_leg_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)

body_frame.set_ang_vel(l_leg_frame, omega2 * l_leg_frame.z)

r_leg_frame.set_ang_vel(body_frame, omega3 * body_frame.z)

# Linear Velocities
# =================

l_ankle.set_vel(inertial_frame, 0)

l_leg_mass_center.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

l_hip.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

body_mass_center.v2pt_theory(l_hip, inertial_frame, body_frame)

r_hip.v2pt_theory(l_hip, inertial_frame, body_frame)

r_leg_mass_center.v2pt_theory(r_hip, inertial_frame, r_leg_frame)

# Mass
# ====

l_leg_mass, body_mass, r_leg_mass = symbols('m_L, m_B, m_R')

# Inertia
# =======

l_leg_inertia, body_inertia, r_leg_inertia = symbols('I_Lz, I_Bz, I_Rz')

l_leg_inertia_dyadic = inertia(l_leg_frame, 0, 0, l_leg_inertia)

l_leg_central_inertia = (l_leg_inertia_dyadic, l_leg_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)

body_central_inertia = (body_inertia_dyadic, body_mass_center)

r_leg_inertia_dyadic = inertia(r_leg_frame, 0, 0, r_leg_inertia)

r_leg_central_inertia = (r_leg_inertia_dyadic, r_leg_mass_center)

# Rigid Bodies
# ============

l_leg = RigidBody('Lower Leg', l_leg_mass_center, l_leg_frame,
                      l_leg_mass, l_leg_central_inertia)

body = RigidBody('Upper Leg', body_mass_center, body_frame,
                      body_mass, body_central_inertia)

r_leg = RigidBody('R_Leg', r_leg_mass_center, r_leg_frame,
                  r_leg_mass, r_leg_central_inertia)

# Gravity
# =======

g = symbols('g')

l_leg_grav_force = (l_leg_mass_center,
                        -l_leg_mass * g * inertial_frame.y)
body_grav_force = (body_mass_center,
                        -body_mass * g * inertial_frame.y)
r_leg_grav_force = (r_leg_mass_center, -r_leg_mass * g * inertial_frame.y)

# Joint Torques
# =============

l_ankle_torque, l_hip_torque, r_hip_torque = dynamicsymbols('T_a, T_k, T_h')

l_leg_torque = (l_leg_frame,
                    l_ankle_torque * inertial_frame.z - l_hip_torque *
                    inertial_frame.z)

body_torque = (body_frame,
                    l_hip_torque * inertial_frame.z - r_hip_torque *
                    inertial_frame.z)

r_leg_torque = (r_leg_frame, r_hip_torque * inertial_frame.z)

# Equations of Motion
# ===================

coordinates = [theta1, theta2, theta3]

speeds = [omega1, omega2, omega3]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [l_leg_grav_force,
         body_grav_force,
         r_leg_grav_force,
         l_leg_torque,
         body_torque,
         r_leg_torque]

bodies = [l_leg, body, r_leg]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# List the symbolic arguments
# ===========================

# Constants
# ---------

constants = [l_leg_length,
             l_leg_com_length,
             l_leg_mass,
             l_leg_inertia,
             hip_width,
             body_com_length,
             body_mass,
             body_inertia,
             r_leg_com_length,
             r_leg_mass,
             r_leg_inertia,
             body_com_height,
             g]

# Time Varying
# ------------

coordinates = [theta1, theta2, theta3]

speeds = [omega1, omega2, omega3]

specified = [l_ankle_torque, l_hip_torque, r_hip_torque]

# Generate RHS Function
# =====================

#right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
#                                        constants, coordinates, speeds,
#                                        specified)

# Specify Numerical Quantities
# ============================

initial_coordinates = deg2rad(90.0) * array([2, 2, 2])

#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([1.035,  # l_leg_length [m]
                             0.58,  # l_leg_com_length [m]
                             23.689,  # l_leg_mass [kg]
                             0.1,  # l_leg_inertia [kg*m^2]
                             0.4,  # hip_width [m]
                             0.2,  # body_com_length
                             32.44,  # body_mass [kg]
                             1.485,  # body_inertia [kg*m^2]
                             0.193,  # r_leg_com_length [m]
                             23.689,  # r_leg_mass [kg]
                             0.1,  # r_leg_inertia [kg*m^2]
                             0.305, #body_com_height
                             9.81],  # acceleration due to gravity [m/s^2]
                           )

#Create dictionaries for the values for the values
zero_speed_dict = dict(zip(speeds, zeros(len(speeds))))
parameter_dict = dict(zip(constants, numerical_constants))
torque_dict = dict(zip([l_ankle_torque], [0]))

#Simplify
#fr_plus_frstar = simplify(fr) + simplify(frstar)
#fr_plus_frstar = simplify(fr_plus_frstar)
#fr_plus_frstar = trigsimp(fr_plus_frstar)

#substitute in values
#subbed_equations = simplify(fr_plus_frstar.subs(zero_speed_dict).subs(torque_dict))
#subbed_value_equations = simplify(subbed_equations.subs(parameter_dict))

#torque_answers_dict = solve(subbed_equations, [l_hip_torque, r_hip_torque])
#solved_dict = solve(subbed_value_equations, [l_hip_torque, r_hip_torque, theta1])

forcing_matrix = kane.forcing

forcing_matrix = simplify(forcing_matrix)

forcing_matrix = simplify(forcing_matrix.subs(zero_speed_dict).subs(parameter_dict).subs(torque_dict))

forcing_solved = solve(forcing_matrix, [l_hip_torque, r_hip_torque, sin(theta1)])

lam_l = lambdify((theta1, theta2, theta3), forcing_solved[l_hip_torque])

lam_r = lambdify((theta1, theta2, theta3), forcing_solved[r_hip_torque])

lam_f = lambdify((theta1, theta2, theta3), forcing_matrix[0])

x = -1.58
y = -3.14
z = -3.14
X = []
Y = []
Z = []

answer_vector = []

threshold = 0.1

while x < 1.58:
  y = -1.57
  z = -1.57
  while y < 3.14:
    z = -1.57
    while z < 3.14:
      lam_sol = lam_f(x,y,z)
      if(lam_sol < threshold and lam_sol > -1*threshold):
        answer_vector.append([lam_sol,lam_l(x,y,z), lam_r(x,y,z), x, y, z])
        X.append(x)
        Y.append(y)
        Z.append(z)
      z = z + 0.01
    y = y + 0.01
  print(x)
  x = x + 0.01

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(X, Y, Z)
#ax.plot_trisurf(X,Y,Z)

ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')

plt.show()
