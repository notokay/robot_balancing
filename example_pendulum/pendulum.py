from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, ones, concatenate
from scipy.integrate import odeint

# Orientations
# ============

theta1, theta2 = dynamicsymbols('theta1, theta2')

inertial_frame = ReferenceFrame('I')

lower_leg_frame = ReferenceFrame('L')

lower_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

upper_leg_frame = ReferenceFrame('U')

upper_leg_frame.orient(lower_leg_frame, 'Axis', (theta2, lower_leg_frame.z))


# Point Locations
# ===============

# Joints
# ------

lower_leg_length, upper_leg_length = symbols('l_L, l_U')

ankle = Point('A')

knee = Point('K')
knee.set_pos(ankle, lower_leg_length * lower_leg_frame.y)

# Center of mass locations
# ------------------------

lower_leg_com_length, upper_leg_com_length = symbols('d_L, d_U')

lower_leg_mass_center = Point('L_o')
lower_leg_mass_center.set_pos(ankle, lower_leg_com_length * lower_leg_frame.y)

upper_leg_mass_center = Point('U_o')
upper_leg_mass_center.set_pos(knee, upper_leg_com_length * upper_leg_frame.y)


# Define kinematical differential equations
# =========================================

omega1, omega2 = dynamicsymbols('omega1, omega2')

time = symbols('t')

kinematical_differential_equations = [omega1 - theta1.diff(time),
                                      omega2 - theta2.diff(time)]

# Angular Velocities
# ==================

lower_leg_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)

upper_leg_frame.set_ang_vel(lower_leg_frame, omega2 * lower_leg_frame.z)


# Linear Velocities
# =================

ankle.set_vel(inertial_frame, 0)

lower_leg_mass_center.v2pt_theory(ankle, inertial_frame, lower_leg_frame)

knee.v2pt_theory(ankle, inertial_frame, lower_leg_frame)

upper_leg_mass_center.v2pt_theory(knee, inertial_frame, upper_leg_frame)

# Mass
# ====

lower_leg_mass, upper_leg_mass = symbols('m_L, m_U')

# Inertia
# =======

lower_leg_inertia, upper_leg_inertia, torso_inertia = symbols('I_Lz, I_Uz, I_Tz')

lower_leg_inertia_dyadic = inertia(lower_leg_frame, 0, 0, lower_leg_inertia)

lower_leg_central_inertia = (lower_leg_inertia_dyadic, lower_leg_mass_center)

upper_leg_inertia_dyadic = inertia(upper_leg_frame, 0, 0, upper_leg_inertia)

upper_leg_central_inertia = (upper_leg_inertia_dyadic, upper_leg_mass_center)

# Rigid Bodies
# ============

lower_leg = RigidBody('Lower Leg', lower_leg_mass_center, lower_leg_frame,
                      lower_leg_mass, lower_leg_central_inertia)

upper_leg = RigidBody('Upper Leg', upper_leg_mass_center, upper_leg_frame,
                      upper_leg_mass, upper_leg_central_inertia)


# Gravity
# =======

g = symbols('g')

lower_leg_grav_force = (lower_leg_mass_center,
                        -lower_leg_mass * g * inertial_frame.y)
upper_leg_grav_force = (upper_leg_mass_center,
                        -upper_leg_mass * g * inertial_frame.y)

# Joint Torques
# =============

ankle_torque, knee_torque = dynamicsymbols('T_a, T_k')

lower_leg_torque = (lower_leg_frame,
                    ankle_torque * inertial_frame.z - knee_torque *
                    inertial_frame.z)

upper_leg_torque = (upper_leg_frame,
                    knee_torque * inertial_frame.z)


# Equations of Motion
# ===================

coordinates = [theta1, theta2]

speeds = [omega1, omega2]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [lower_leg_grav_force,
         upper_leg_grav_force,
         lower_leg_torque,
         upper_leg_torque]

bodies = [lower_leg, upper_leg]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# List the symbolic arguments
# ===========================

# Constants
# ---------

constants = [lower_leg_length,
             lower_leg_com_length,
             lower_leg_mass,
             lower_leg_inertia,
             upper_leg_length,
             upper_leg_com_length,
             upper_leg_mass,
             upper_leg_inertia,
             g]

# Time Varying
# ------------

coordinates = [theta1, theta2]

speeds = [omega1, omega2]

specified = [ankle_torque, knee_torque]

# Generate RHS Function
# =====================

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)






