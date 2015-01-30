from sympy import symbols, Matrix, simplify,trigsimp, sin, cos
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod, inertia_of_point_mass
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, ones, concatenate
from scipy.integrate import odeint
#from sympy import *
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


# Orientations
# ============

theta1_dp1, theta2_dp1, theta1_dp2, theta2_dp2 = dynamicsymbols('theta_1dp1, theta2_dp1, theta_1dp2, theta2_dp2')

inertial_frame_dp1 = ReferenceFrame('I_dp1')
inertial_frame_dp2 = ReferenceFrame('I_dp2')

one_frame_dp1 = ReferenceFrame('1_dp1')
one_frame_dp2 = ReferenceFrame('2_dp2')

one_frame_dp1.orient(inertial_frame_dp1, 'Axis', (theta1_dp1, inertial_frame_dp1.z))
one_frame_dp2.orient(inertial_frame_dp2, 'Axis', (theta1_dp2, inertial_frame_dp2.z))

two_frame_dp1 = ReferenceFrame('2_dp1')
two_frame_dp2 = ReferenceFrame('2_dp2')

two_frame_dp1.orient(one_frame_dp1, 'Axis', (theta2_dp1, one_frame_dp1.z))
two_frame_dp2.orient(one_frame_dp2, 'Axis', (theta2_dp2, one_frame_dp2.z))

# Point Locations
# ===============

# Joints
# ------

one_length_dp1, one_length_dp2, two_length_dp2 = symbols('l_1dp1, l_1dp2, l_2dp2')

one_dp1 = Point('1_dp1')
one_dp2 = Point('1_dp2')

two_dp1 = Point('2_dp1')
two_dp2 = Point('2_dp2')
two_dp1.set_pos(one_dp1, one_length_dp1 * one_frame_dp1.y)
two_dp2.set_pos(one_dp2, one_length_dp2 * one_frame_dp2.y)

three_dp2 = Point('3_dp2')
three_dp2.set_pos(two_dp2, two_length_dp2 * two_frame_dp2.y)

# Center of mass locations
# ------------------------

one_com_xdp1, one_com_ydp1, two_com_xdp1, two_com_ydp1 = dynamicsymbols('1_comx_dp1, 1_comy_dp1, 2_comx_dp1, 2_comy_dp1')

one_mass_center_dp1 = Point('1_odp1')
one_mass_center_dp1.set_pos(one_dp1, one_length_dp1*one_frame_dp1.y)

two_mass_center_dp1 = Point('2_odp1')
two_mass_center_dp1.set_pos(one_dp1, two_com_xdp1*inertial_frame_dp1.x + two_com_ydp1*inertial_frame_dp1.y)

# Define kinematical differential equations
# =========================================

omega1_dp1, omega2_dp1, omega1_dp2, omega2_dp2 = dynamicsymbols('omega_1dp1, omega_2dp1, omega_1dp2, omega_2dp2')

time = symbols('t')

kinematical_differential_equations_dp1 = [omega1_dp1 - theta1_dp1.diff(time),
                                          omega2_dp1 - theta2_dp2.diff(time)]

kinematical_differential_equations_dp2 = [omega1_dp2 - theta1_dp2.diff(time),
                                          omega2_dp2 - theta2_dp2.diff(time)]

# Angular Velocities
# ==================

one_frame_dp1.set_ang_vel(inertial_frame_dp1, omega1_dp1 * inertial_frame_dp1.z)
one_frame_dp1.ang_vel_in(inertial_frame_dp1)

one_frame_dp2.set_ang_vel(inertial_frame_dp2, omega1_dp2 * inertial_frame_dp2.z)
one_frame_dp2.ang_vel_in(inertial_frame_dp2)

two_frame_dp1.set_ang_vel(one_frame_dp1, omega2_dp1 * one_frame_dp1.z - omega1_dp2*one_frame_dp1.z)
two_frame_dp1.ang_vel_in(inertial_frame_dp1)
two_frame_dp2.set_ang_vel(one_frame_dp2, omega2_dp2 * one_frame_dp2.z)
two_frame_dp2.ang_vel_in(inertial_frame_dp2)

# Linear Velocities
# =================

one_dp1.set_vel(inertial_frame_dp1, 0)
one_dp2.set_vel(inertial_frame_dp2, two_dp1.vel)

one_mass_center_dp1.v2pt_theory(one_dp1, inertial_frame_dp1, one_frame_dp1)
one_mass_center_dp1.vel(inertial_frame_dp1)
one_mass_center_dp2.v2pt_theory(one_dp2, inertial_frame_dp2, one_frame_dp2)
one_mass_center_dp2.vel(inertial_frame_dp2)

two_dp1.v2pt_theory(one_dp1, inertial_frame_dp1, one_frame_dp1)
two_dp1.vel(inertial_frame_dp1)
two_dp2.v2pt_theory(one_dp2, inertial_frame_dp2, one_frame_dp2)
two_dp2.vel(inertial_frame_dp2)

two_mass_center_dp1.v2pt_theory(two_dp1, inertial_frame_dp1, two_frame_dp1)
two_mass_center_dp2.v2pt_theory(two_dp2, inertial_frame_dp2, two_frame_dp2)

# Mass
# ====

a_mass, b_mass, c_mass = symbols('m_a, m_b, m_c')

# Inertia
# =======

rotI = lambda I, f: Matrix([j & I & i for i in f for j in 
f]).reshape(3,3)

one_inertia_dyadic_dp1 = inertia(one_frame_dp1, 0, 0, rotI(inertia_of_point_mass(a_mass, one_mass_center_dp1.pos_from(one_dp1), one_frame_dp1), one_frame_dp1)[8])

one_central_inertia_dp1 = (one_inertia_dyadic_dp1, one_mass_center_dp1)

two_inertia_dyadic_dp1 = inertia(two_frame_dp1, 0, 0, rotI(inertia_of_point_mass(b_mass+c_mass, two_mass_center_dp1.pos_from(one_dp1), one_frame_dp1), one_frame_dp1)[8])

two_central_inertia_dp1 = (two_inertia_dyadic_dp1, two_mass_center_dp1)

# Rigid Bodies
# ============

oneB_dp1 = RigidBody('One_dp1', one_mass_center_dp1, one_frame_dp1,
                      a_mass, one_central_inertia_dp1)

twoB_dp1 = RigidBody('Two_dp1', two_mass_center_dp1, two_frame_dp1,
                      b_mass+c_mass, two_central_inertia_dp1)

#Defines the linkages as particles
oneP_dp2 = Particle('twoP', one_dp2 , b_mass)
twoP_dp2 = Particle('threeP', two_dp2, c_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')

one_grav_force_vector_dp1 = -1*a_mass*g*inertial_frame_dp1.y
one_grav_force_dp1 = (one_dp1, one_grav_force_vector_dp1)
two_grav_force_vector_dp1 = -1*(b_mass+c_mass)*g*inertial_frame_dp1.y
two_grav_force_dp1 = (two_dp1, two_grav_force_vector_dp1)

one_grav_force_vector_dp2 = -1*b_mass*g*inertial_frame_dp2.y
one_grav_force_dp2 = (one_dp2, one_grav_force_vector_dp2)
two_grav_force_vector_dp2 = -1*(c_mass)*g*inertial_frame_dp2.y
two_grav_force_dp2 = (two_dp2, two_grav_force_vector_dp2)

# Joint Torques
# =============

one_torque_dp1, two_torque_dp2, one_torque_dp2, two_torque_dp2 = dynamicsymbols('T_1dp1, T_2dp2, T_1dp2, T_2dp2')

one_link_torque_dp1 = (one_frame_dp1,
                       one_torque_dp1 * inertial_frame_dp1.z - two_torque_dp1 * inertial_frame_dp1.z)

two_link_torque_dp1 = (two_frame_dp1,
                       two_torque_dp1 * inertial_frame_dp1.z + one_torque_dp2*inertial_frame_dp1.z)

one_link_torque_dp2 = (one_frame_dp2, 
                       one_torque_dp2 * inertial_frame_dp2.z - two_torque_dp2 * inertial_frame_dp2.z)

two_link_torque_dp2 = (two_frame_dp2,
                       two_torque_dp2 * inertial_frame_dp2.z)

# Equations of Motion
# ===================

coordinates_dp1 = [theta1_dp1, theta2_dp1]
coordinates_dp2 = [theta1_dp2, theta2_dp2]

speeds_dp1 = [omega1_dp1, omega2_dp1]
speeds_dp2 = [omega1_dp2, omega2_dp2]

kane_dp1 = KanesMethod(inertial_frame_dp1,
                       coordinates_dp1,
                       speeds_dp1,
                       kinematical_differential_equations_dp1)
kane_dp2 = KanesMethod(inertial_frame_dp2,
                       coordinates_dp2,
                       speeds_dp2,
                       kinematic_differential_equations_dp2)

loads_dp1 = [one_grav_force_dp1,
             two_grav_force_dp1,
             one_link_torque_dp1,
             two_link_torque_dp1]
loads_dp2 = [one_grav_force_dp2,
             two_grav_force_dp2,
             two_link_torque_dp2,
             two_link_torque_dp2]

bodies_dp1 = [oneB_dp1, twoB_dp1]
particles_dp2 = [oneP_dp2, twoP_dp2]

fr_dp1, frstar_dp1 = kane_dp1.kanes_equations(loads, bodies)

mass_matrix_dp1 = kane_dp1.mass_matrix
forcing_vector_dp1 = kane_dp1.forcing

mass_matrix_dp2 = kane_dp2.mass_matrix
forcing_vector_dp2 = kane_dp2.forcing




