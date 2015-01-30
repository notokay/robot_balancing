from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from pydy.codegen.code import generate_ode_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
a_frame = ReferenceFrame('A')
base_frame = ReferenceFrame('Base')
#Sets up symbols for joint angles
theta_a = dynamicsymbols('theta_a')
theta_base = dynamicsymbols('theta_base')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
a_frame.orient(inertial_frame, 'Axis', (theta_a, inertial_frame.z))
base_frame.orient(inertial_frame, 'Axis', (theta_base, inertial_frame.z))

#Sets up points for the joints and places them relative to each other
A = Point('A')
a_length = symbols('l_a')
B = Point('B')
B.set_pos(A, a_length*a_frame.y)

#Sets up the angular velocities
omega_base, omega_a = dynamicsymbols('omega_b, omega_a')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega_a - theta_a.diff()]

#Sets up the rotational axes of the angular velocities
a_frame.set_ang_vel(inertial_frame, omega_base*inertial_frame.z + omega_a*inertial_frame.z)

#Sets up the linear velocities of the points on the linkages
base_vel_x, base_vel_y = dynamicsymbols('v_bx, v_by')
A.set_vel(inertial_frame, base_vel_x*base_frame.x + base_vel_y*base_frame.y)
B.v2pt_theory(A, inertial_frame, a_frame)

#Sets up the masses of the linkages
a_mass = symbols('m_a')

#Defines the linkages as particles
bP = Particle('bP', B, a_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
b_grav_force_vector = -1*a_mass*g*inertial_frame.y
b_grav_force = (B, b_grav_force_vector)

#Sets up joint torques
a_torque = dynamicsymbols('T_a')
a_torque_vector = a_torque*inertial_frame.z
a_link_torque = (a_frame, a_torque_vector)

#Generalized coordinates
coordinates = [theta_a]

#Generalized speeds
speeds = [omega_a]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [b_grav_force,
         a_link_torque]
bodies = [bP]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# Constants
# ---------

constants = [a_length,
             a_mass,
             g]

specified = [a_torque]

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)

