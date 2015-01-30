from sympy import symbols, Matrix, simplify,trigsimp, sin, cos, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod, inertia_of_point_mass
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, ones, concatenate
from scipy.integrate import odeint
#from sympy import *
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


# Orientations
# ============

theta1, theta2 = dynamicsymbols('theta1, theta2')

inertial_frame = ReferenceFrame('I')

one_frame = ReferenceFrame('1')

one_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

two_frame = ReferenceFrame('2')

two_frame.orient(one_frame, 'Axis', (theta2, one_frame.z))


# Point Locations
# ===============

# Joints
# ------

one_length = symbols('l_1')

one = Point('1')

two = Point('2')
two.set_pos(one, one_length * one_frame.y)

# Center of mass locations
# ------------------------

one_com_x, one_com_y, two_com_x, two_com_y = dynamicsymbols('1_comx, 1_comy, 2_comx, 2_comy')

one_mass_center = Point('1_o')
one_mass_center.set_pos(one, one_com_x*inertial_frame.x + one_com_y*inertial_frame.y)

two_mass_center = Point('2_o')
two_mass_center.set_pos(two, two_com_x*inertial_frame.x + two_com_y*inertial_frame.y)


# Define kinematical differential equations
# =========================================

omega1, omega2 = dynamicsymbols('omega1, omega2')

time = symbols('t')

kinematical_differential_equations = [omega1 - theta1.diff(time),
                                      omega2 - theta2.diff(time)]

# Angular Velocities
# ==================

one_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)
one_frame.ang_vel_in(inertial_frame)
two_frame.set_ang_vel(one_frame, omega2 * one_frame.z)
two_frame.ang_vel_in(inertial_frame)

# Linear Velocities
# =================

one.set_vel(inertial_frame, 0)

one_mass_center.v2pt_theory(one, inertial_frame, one_frame)
one_mass_center.vel(inertial_frame)
two.v2pt_theory(one, inertial_frame, one_frame)
two.vel(inertial_frame)
two_mass_center.v2pt_theory(two, inertial_frame, two_frame)

# Mass
# ====

one_mass, two_mass = symbols('m_1, m_2')

# Inertia
# =======

rotI = lambda I, f: Matrix([j & I & i for i in f for j in 
f]).reshape(3,3)

one_inertia_dyadic = inertia(one_frame, 0, 0, rotI(inertia_of_point_mass(one_mass, one_mass_center.pos_from(one), one_frame), one_frame)[8])

one_central_inertia = (one_inertia_dyadic, one_mass_center)

two_inertia_dyadic = inertia(two_frame, 0, 0, rotI(inertia_of_point_mass(two_mass, two_mass_center.pos_from(one), one_frame), one_frame)[8])

two_central_inertia = (two_inertia_dyadic, two_mass_center)

# Rigid Bodies
# ============

oneB = RigidBody('One', one_mass_center, one_frame,
                      one_mass, one_central_inertia)

twoB = RigidBody('Upper Leg', two_mass_center, two_frame,
                      two_mass, two_central_inertia)


# Gravity
# =======

g = symbols('g')

one_grav_force = (one_mass_center,
                        -one_mass * g * inertial_frame.y)
two_grav_force = (two_mass_center,
                        -two_mass * g * inertial_frame.y)

# Joint Torques
# =============

one_torque, two_torque = dynamicsymbols('T_1, T_2')

one_link_torque = (one_frame,
                    one_torque * inertial_frame.z - two_torque *
                    inertial_frame.z)

two_link_torque = (two_frame,
                    two_torque * inertial_frame.z)


# Equations of Motion
# ===================

coordinates = [theta1, theta2]

speeds = [omega1, omega2]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [one_grav_force,
         two_grav_force,
         one_link_torque,
         two_link_torque]

bodies = [oneB, twoB]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# List the symbolic arguments
# ===========================

# Constants
# ---------

constants = [one_length,
             one_com_x,
             one_com_y,
             one_mass,
             two_com_x,
             two_com_y,
             two_mass,
             g]

# Time Varying
# ------------

coordinates = [theta1, theta2]

speeds = [omega1, omega2]

specified = [one_torque, two_torque]

forcing = kane.forcing

mass_matrix = kane.mass_matrix

zero_omega = dict(zip(speeds, [0,0]))

torques = Matrix(specified)

g_terms = simplify(forcing.subs(zero_omega) - torques)

g_terms_1 = g_terms[0]
g_terms_2 = g_terms[1]

coriolis = simplify(forcing - g_terms - torques)

r_1i = simplify(one_frame.dcm(inertial_frame))
t_1i = simplify(one_mass_center.pos_from(one).express(inertial_frame).to_matrix(inertial_frame))

t_2i = simplify(two_mass_center.pos_from(one).express(inertial_frame).to_matrix(inertial_frame))

time = symbols('t')

alpha1, alpha2 = dynamicsymbols('a_1, a_2')
accelerations = [alpha1, alpha2]

thetadot_omega_dict = dict(zip([theta1.diff(time), theta2.diff(time)], [omega1, omega2]))
omegadot_alpha_dict = dict(zip([omega1.diff(time), omega2.diff(time)], [alpha1, alpha2]))

com = (t_1i*one_mass + t_2i*two_mass)/(one_mass+two_mass)

v_t_1i = one_mass_center.vel(inertial_frame).express(inertial_frame).to_matrix(inertial_frame)
v_t_2i = two_mass_center.vel(inertial_frame).express(inertial_frame).to_matrix(inertial_frame)

com_dot = (v_t_1i*one_mass + v_t_2i*two_mass)/(one_mass+two_mass)

com_ddot = com_dot.diff(time)
com_ddot = com_ddot.subs(thetadot_omega_dict).subs(omegadot_alpha_dict).as_mutable()
des_com_ang_acc_x, des_com_ang_acc_y = dynamicsymbols('c_x, c_y')
com_ddot[0] = com_ddot[0] - des_com_ang_acc_x
com_ddot[1] = com_ddot[1] - des_com_ang_acc_y

desired_ang_acc_from_com_acc = solve(com_ddot, [alpha1, alpha2])

des_t1_acc = desired_ang_acc_from_com_acc.get(alpha1)
des_t2_acc = desired_ang_acc_from_com_acc.get(alpha2)

one_com_x_dot, one_com_y_dot, two_com_x_dot, two_com_y_dot = dynamicsymbols('1_comxdot, 1_comydot, 2_comxdot, 2_comydot')

comdot_comdot_dict = dict(zip([one_com_x.diff(time), one_com_y.diff(time), two_com_x.diff(time), two_com_y.diff(time)], [one_com_x_dot, one_com_y_dot, two_com_x_dot, two_com_y_dot]))

des_t1_acc = des_t1_acc.subs(comdot_comdot_dict)
des_t2_acc = des_t2_acc.subs(comdot_comdot_dict)

acc = Matrix(accelerations)

inverse_dynamics = (mass_matrix*acc - g_terms + coriolis).subs(comdot_comdot_dict)
