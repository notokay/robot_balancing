from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from pydy.codegen.code import generate_ode_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
s_frame = ReferenceFrame('S')

#Sets up symbols for joint angles
theta_s = dynamicsymbols('theta_s')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
s_frame.orient(inertial_frame, 'Axis', (theta_s, inertial_frame.z))

#Sets up points for the joints and places them relative to each other
S = Point('S')
s_length = symbols('l_s')
T = Point('T')
T.set_pos(S, s_length*s_frame.y)

#Sets up the angular velocities
omega_s = dynamicsymbols('omega_s')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega_s - theta_s.diff()]

#Sets up the rotational axes of the angular velocities
s_frame.set_ang_vel(inertial_frame, omega_s*inertial_frame.z)

#Sets up the linear velocities of the points on the linkages
S.set_vel(inertial_frame, 0)
T.v2pt_theory(S, inertial_frame, s_frame)

#Sets up the masses of the linkages
s_mass = symbols('m_s')

#Defines the linkages as particles
tP = Particle('tP', T, s_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
t_grav_force_vector = -1*s_mass*g*inertial_frame.y
t_grav_force = (T, t_grav_force_vector)

#Sets up joint torques
s_torque = dynamicsymbols('T_s')
s_torque_vector = s_torque*inertial_frame.z
s_link_torque = (s_frame, s_torque_vector)

#Generalized coordinates
coordinates = [theta_s]

#Generalized speeds
speeds = [omega_s]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [t_grav_force,
         s_link_torque]
bodies = [tP]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix
forcing = kane.forcing[0]

# Constants
# ---------

constants = [s_length,
             s_mass,
             g]
time = symbols('t')
omega_s = dynamicsymbols('omega_s')
thetadot_omega_dict = dict(zip([theta_s.diff(time)], [omega_s]))
alpha_s = dynamicsymbols('alpha_s')
omegadot_alpha_dict = dict(zip([omega_s.diff(time)], [alpha_s]))

forces = forcing - s_torque

ri = s_frame.dcm(inertial_frame)

com = T.pos_from(S).express(inertial_frame).to_matrix(inertial_frame)
com_dot = com.diff(time).subs(thetadot_omega_dict)
com_ddot = com_dot.diff(time).subs(thetadot_omega_dict).subs(omegadot_alpha_dict)
