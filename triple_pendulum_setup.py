from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy import symbols
from numpy import array,  zeros
import numpy as np

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

# Specify Numerical Quantities
# ============================

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([1.035,  # l_leg_length [m]
                             0.58,  # l_leg_com_length [m]
                             23.689,  # l_leg_mass [kg]
                             0.1,  # l_leg_inertia [kg*m^2]
                             0.4,  # hip_width [m]
                             0.2,  # body_com_length
                             32.44,  # body_mass [kg]
                             1.485,  # body_inertia [kg*m^2]
                             0.58,  # r_leg_com_length [m]
                             23.689,  # r_leg_mass [kg]
                             0.1,  # r_leg_inertia [kg*m^2]
                             0.305, #body_com_height
                             9.81],  # acceleration due to gravity [m/s^2]
                           )

parameter_dict = dict(zip(constants, numerical_constants))
