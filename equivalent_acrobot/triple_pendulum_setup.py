from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()

time = symbols('t')
#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
a_frame = ReferenceFrame('A')
b_frame = ReferenceFrame('B')
c_frame = ReferenceFrame('C')

#Sets up symbols for joint angles
theta_a, theta_b, theta_c = dynamicsymbols('theta_a, theta_b, theta_c')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
a_frame.orient(inertial_frame, 'Axis', (theta_a, inertial_frame.z))
b_frame.orient(a_frame, 'Axis', (theta_b, a_frame.z))
c_frame.orient(b_frame, 'Axis', (theta_c, b_frame.z))

#Sets up points for the joints and places them relative to each other
A = Point('A')
a_length = symbols('l_a')
B = Point('B')
B.set_pos(A, a_length*a_frame.y)
C = Point('C')
b_length = symbols('l_b')
C.set_pos(B, b_length*b_frame.y)
D = Point('D')
c_length = symbols('l_c')
D.set_pos(C, c_length*c_frame.y)

#Sets up the angular velocities
omega_a, omega_b, omega_c = dynamicsymbols('omega_a, omega_b, omega_c')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega_a - theta_a.diff(),
                                    omega_b - theta_b.diff(),
                                    omega_c - theta_c.diff()]

#Sets up the rotational axes of the angular velocities
a_frame.set_ang_vel(inertial_frame, omega_a*inertial_frame.z)
a_frame.ang_vel_in(inertial_frame)
b_frame.set_ang_vel(a_frame, omega_b*inertial_frame.z)
b_frame.ang_vel_in(inertial_frame)
c_frame.set_ang_vel(b_frame, omega_c*inertial_frame.z)
c_frame.ang_vel_in(inertial_frame)

#Sets up the linear velocities of the points on the linkages
A.set_vel(inertial_frame, 0)
B.v2pt_theory(A, inertial_frame, a_frame)
B.vel(inertial_frame)
C.v2pt_theory(B, inertial_frame, b_frame)
C.vel(inertial_frame)
D.v2pt_theory(C, inertial_frame, c_frame)
D.vel(inertial_frame)

#Sets up the masses of the linkages
a_mass, b_mass, c_mass = symbols('m_a, m_b, m_c')

#Defines the linkages as particles
bP = Particle('bP', B, a_mass)
cP = Particle('cP', C, b_mass)
dP = Particle('dP', D, c_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
b_grav_force_vector = -1*a_mass*g*inertial_frame.y
b_grav_force = (B, b_grav_force_vector)
c_grav_force_vector = -1*b_mass*g*inertial_frame.y
c_grav_force = (C, c_grav_force_vector)
d_grav_force_vector = -1*c_mass*g*inertial_frame.y
d_grav_force = (D, d_grav_force_vector)

#Sets up joint torques
a_torque, b_torque, c_torque = dynamicsymbols('T_a, T_b, T_c')
a_torque_vector = a_torque*inertial_frame.z - b_torque*inertial_frame.z
a_link_torque = (a_frame, a_torque_vector)

b_torque_vector = b_torque*inertial_frame.z - c_torque*inertial_frame.z
b_link_torque = (b_frame, b_torque_vector)

c_torque_vector = c_torque*inertial_frame.z
c_link_torque = (c_frame, c_torque_vector)

#Generalized coordinates
coordinates = [theta_a, theta_b, theta_c]

#Generalized speeds
speeds = [omega_a, omega_b, omega_c]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [b_grav_force,
         c_grav_force,
         d_grav_force,
         a_link_torque,
         b_link_torque,
         c_link_torque]
bodies = [bP, cP, dP]

fr, frstar = kane.kanes_equations(loads, bodies)
frplusfrstar = simplify(trigsimp(fr + frstar))
mass_matrix_full = kane.mass_matrix_full

forcing_vector_full = kane.forcing_full

constants = [a_length,
             a_mass,
             b_length,
             b_mass,
             c_length,
             c_mass,
             g]
#Specified contains the matrix for the input torques
specified = [a_torque, b_torque, c_torque]

#Specifies numerical constants for inertial/mass properties
#Robot Params
#numerical_constants = array([1.035,  # leg_length[m]
#                             36.754, # leg_mass[kg]
#			     0.85, # body_length[m]
#                             91.61,  # body_mass[kg]
#                             9.81]    # acceleration due to gravity [m/s^2]
#                             )
numerical_constants = array([1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             9.81])

#Set input torques to 0
numerical_specified = zeros(3)

parameter_dict = dict(zip(constants, numerical_constants))

ke_energy = simplify(kinetic_energy(inertial_frame, bP, cP, dP))

bP.set_potential_energy(a_mass*g*a_length*cos(theta_a))

cP.set_potential_energy(b_mass*g*(a_length*cos(theta_a)+b_length*cos(theta_a+theta_b)))

dP.set_potential_energy(c_mass*g*(a_length*cos(theta_a)+b_length*cos(theta_a+theta_b)+c_length*cos(theta_a+theta_b+theta_c)))

pe_energy = simplify(potential_energy(bP, cP, dP).subs(parameter_dict))

forcing = simplify(kane.forcing)

mass_matrix = simplify(kane.mass_matrix)

zero_omega = dict(zip(speeds, zeros(3)))

torques = Matrix(specified)

g_terms = forcing.subs(zero_omega) - torques

g_terms_a = g_terms[0]
g_terms_b = g_terms[1]
g_terms_c = g_terms[2]

ang_vel = Matrix(speeds)

coriolis = simplify(forcing - g_terms - torques)

r_ai = a_frame.dcm(inertial_frame).as_mutable()
t_ai = simplify(B.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))

r_bi = simplify(b_frame.dcm(inertial_frame)).as_mutable()
t_bi = simplify(C.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))
t_ba = simplify(C.pos_from(B).express(inertial_frame).to_matrix(inertial_frame))
t_ba_in_a = simplify(C.pos_from(B).express(a_frame).to_matrix(a_frame))
tf_bi = r_bi.row_join(t_bi)

r_ci = simplify(c_frame.dcm(inertial_frame)).as_mutable()
t_ci = simplify(D.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))
t_ca = simplify(D.pos_from(B).express(inertial_frame).to_matrix(inertial_frame))
t_ca_in_a = simplify(D.pos_from(B).express(a_frame).to_matrix(a_frame))

thetadot_omega_dict = dict(zip([theta_a.diff(time), theta_b.diff(time), theta_c.diff(time)], speeds))

com = simplify(t_ai*a_mass + t_bi*b_mass + t_ci*c_mass)/(a_mass+b_mass+c_mass)
com_dot = com.diff(time).subs(thetadot_omega_dict)
com_bc = simplify(t_ba*b_mass + t_ca*c_mass)/(b_mass+c_mass)
com_bc_in_a = simplify(t_ba_in_a*b_mass + t_ca_in_a*c_mass)/(b_mass+c_mass)
com_bc_dot_in_a = com_bc_in_a.diff(time)
com_bc_dot_in_a = com_bc_dot_in_a.subs(thetadot_omega_dict)
com_bc_ddot_in_a = com_bc_dot_in_a.diff(time).subs(thetadot_omega_dict)
com_bc_dot = com_bc.diff(time)
com_bc_dot = com_bc.subs(thetadot_omega_dict)
com_a = t_ai
com_a_dot = com_a.diff(time).subs(thetadot_omega_dict)

