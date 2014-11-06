from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod, kinetic_energy
from sympy import symbols, cos, sin
from numpy import array,  zeros
import numpy as np

# Orientations
# ============

theta1, theta2, theta3, theta4 = dynamicsymbols('theta1, theta2, theta3, theta4')

inertial_frame = ReferenceFrame('I')

l_leg_frame = ReferenceFrame('L')

l_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

crotch_frame = ReferenceFrame('C')

crotch_frame.orient(l_leg_frame, 'Axis', (theta2, l_leg_frame.z))

r_leg_frame = ReferenceFrame('R')

r_leg_frame.orient(crotch_frame, 'Axis', (theta3, crotch_frame.z))

body_frame = ReferenceFrame('B')

body_frame.orient(crotch_frame, 'Axis', (theta4, crotch_frame.z))

# Point Locations
# ===============

# Joints
# ------

l_leg_length, hip_width, body_height = symbols('l_L, h_W, b_H')

l_ankle = Point('LA')

l_hip = Point('LH')
l_hip.set_pos(l_ankle, l_leg_length * l_leg_frame.y)

r_hip = Point('RH')
r_hip.set_pos(l_hip, hip_width * crotch_frame.x)

body_middle = Point('B_m')
body_middle.set_pos(l_hip, (hip_width/2)*crotch_frame.x)

waist = Point('W')
waist.set_pos(body_middle, body_height*crotch_frame.y)

# Center of mass locations
# ------------------------

l_leg_com_length, r_leg_com_length, crotch_com_height, body_com_height = symbols('d_L, d_R, d_C, d_B')

l_leg_mass_center = Point('LL_o')
l_leg_mass_center.set_pos(l_ankle, l_leg_com_length * l_leg_frame.y)

crotch_mass_center = Point('C_o')
crotch_mass_center.set_pos(body_middle, crotch_com_height*crotch_frame.y)

body_mass_center = Point('B_o')
body_mass_center.set_pos(waist, body_com_height*body_frame.y)

r_leg_mass_center = Point('RL_o')
r_leg_mass_center.set_pos(r_hip, -1*r_leg_com_length * r_leg_frame.y)

# Define kinematical differential equations
# =========================================

omega1, omega2, omega3, omega4 = dynamicsymbols('omega1, omega2, omega3, omega4')

time = symbols('t')

kinematical_differential_equations = [omega1 - theta1.diff(time),
                                      omega2 - theta2.diff(time),
                                      omega3 - theta3.diff(time),
                                      omega4 - theta4.diff(time)]

# Angular Velocities
# ==================

l_leg_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)

crotch_frame.set_ang_vel(l_leg_frame, omega2 * l_leg_frame.z)

body_frame.set_ang_vel(crotch_frame, omega4*crotch_frame.z)

r_leg_frame.set_ang_vel(crotch_frame, omega3 * crotch_frame.z)

# Linear Velocities
# =================

l_ankle.set_vel(inertial_frame, 0)

l_leg_mass_center.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

l_hip.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

crotch_mass_center.v2pt_theory(l_hip, inertial_frame, crotch_frame)

waist.v2pt_theory(l_hip, inertial_frame, crotch_frame)

body_mass_center.v2pt_theory(waist, inertial_frame, body_frame)

r_hip.v2pt_theory(l_hip, inertial_frame, crotch_frame)

r_leg_mass_center.v2pt_theory(r_hip, inertial_frame, r_leg_frame)

# Mass
# ====

l_leg_mass, crotch_mass, body_mass, r_leg_mass = symbols('m_L, m_C, m_B, m_R')

# Inertia
# =======

l_leg_inertia, crotch_inertia, body_inertia, r_leg_inertia = symbols('I_Lz, I_Cz, I_Bz, I_Rz')

l_leg_inertia_dyadic = inertia(l_leg_frame, 0, 0, l_leg_inertia)

l_leg_central_inertia = (l_leg_inertia_dyadic, l_leg_mass_center)

crotch_inertia_dyadic = inertia(crotch_frame, 0, 0, crotch_inertia)

crotch_central_inertia = (l_leg_inertia_dyadic, crotch_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)

body_central_inertia = (body_inertia_dyadic, body_mass_center)

r_leg_inertia_dyadic = inertia(r_leg_frame, 0, 0, r_leg_inertia)

r_leg_central_inertia = (r_leg_inertia_dyadic, r_leg_mass_center)

# Rigid Bodies
# ============

l_leg = RigidBody('Left Leg', l_leg_mass_center, l_leg_frame,
                      l_leg_mass, l_leg_central_inertia)

crotch = RigidBody('Crotch', crotch_mass_center, crotch_frame, 
                   crotch_mass, crotch_central_inertia)

body = RigidBody('Body', body_mass_center, body_frame,
                      body_mass, body_central_inertia)

r_leg = RigidBody('Right Leg', r_leg_mass_center, r_leg_frame,
                  r_leg_mass, r_leg_central_inertia)

# Gravity
# =======

g = symbols('g')

l_leg_grav_force = (l_leg_mass_center,
                        -l_leg_mass * g * inertial_frame.y)

crotch_grav_force = (crotch_mass_center, 
                        -crotch_mass * g * inertial_frame.y)

body_grav_force = (body_mass_center,
                        -body_mass * g * inertial_frame.y)

r_leg_grav_force = (r_leg_mass_center, 
                        -r_leg_mass * g * inertial_frame.y)

# Joint Torques
# =============

l_ankle_torque, l_hip_torque, waist_torque, r_hip_torque = dynamicsymbols('T_a, T_l, T_w, T_r')

l_leg_torque = (l_leg_frame,
                    l_ankle_torque * inertial_frame.z - l_hip_torque *
                    inertial_frame.z)

crotch_torque = (crotch_frame, l_hip_torque*inertial_frame.z - r_hip_torque*
                     inertial_frame.z - waist_torque*inertial_frame.z)

body_torque = (body_frame, waist_torque * inertial_frame.z)
              
r_leg_torque = (r_leg_frame, r_hip_torque * inertial_frame.z)

# Equations of Motion
# ===================

coordinates = [theta1, theta2, theta3, theta4]

speeds = [omega1, omega2, omega3, omega4]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [l_leg_grav_force,
         crotch_grav_force,
         body_grav_force,
         r_leg_grav_force,
         l_leg_torque,
         crotch_torque,
         body_torque,
         r_leg_torque]

bodies = [l_leg, crotch, body, r_leg]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# Energies
# ========
ke_lleg = kinetic_energy(inertial_frame, l_leg)
l_leg.set_potential_energy(l_leg_mass*g*l_leg_com_length*cos(theta1))

ke_crotch = kinetic_energy(inertial_frame, crotch) - ke_lleg
crotch.set_potential_energy(crotch_mass*g*(l_leg_length*cos(theta1) + (hip_width/2)*sin(theta1+theta2) + crotch_com_height*sin(theta1+theta2+1.57)))

ke_body = kinetic_energy(inertial_frame, body) - ke_crotch - ke_lleg
body.set_potential_energy(body_mass*g*(l_leg_length*cos(theta1) + (hip_width/2)*sin(theta1+theta2) + body_height*sin(theta1+theta2+1.57) + body_com_height*cos(theta1+theta2+1.57+theta4)))

ke_rleg = kinetic_energy(inertial_frame, r_leg)
r_leg.set_potential_energy(r_leg_mass*g*(l_leg_length*cos(theta1) + hip_width*sin(theta1+theta2) + -1*r_leg_com_length*cos(theta1+theta2+theta3)))


# List the symbolic arguments
# ===========================

# Constants
# ---------

constants = [l_leg_length,
             l_leg_com_length,
             l_leg_mass,
             l_leg_inertia,
             hip_width,
             crotch_com_height,
             crotch_mass,
             crotch_inertia,
             body_height,
             body_com_height,
             body_mass,
             body_inertia,
             r_leg_com_length,
             r_leg_mass,
             r_leg_inertia,
             g]

# Time Varying
# ------------

coordinates = [theta1, theta2, theta3, theta4]

speeds = [omega1, omega2, omega3, omega4]

specified = [l_ankle_torque, l_hip_torque, r_hip_torque, waist_torque]

# Specify Numerical Quantities
# ============================

numerical_constants = array([0.8,      # l_leg_length [m]
                             0.4,      # l_leg_com_length [m]
                             20.0,     # l_leg_mass [kg]
                             1.0,  # l_leg_inertia [kg*m^2]
                             0.25,     # hip_width [m]
                             0.05,     # crotch_com_height [m]
                             20.0,     # crotch_mass [kg]
                             0.1,      # crotch_inertia [kg*m^2]
                             0.15,     # body_height
                             0.5,      # body_com_height
                             50.0,     # body_mass [kg]
                             4.0,    # body_inertia [kg*m^2]
                             0.4,      # r_leg_com_length [m]
                             20.0,     # r_leg_mass [kg]
                             1.0,  # r_leg_inertia [kg*m^2]
                             9.81],    # acceleration due to gravity [m/s^2]
                           )


parameter_dict = dict(zip(constants, numerical_constants))
