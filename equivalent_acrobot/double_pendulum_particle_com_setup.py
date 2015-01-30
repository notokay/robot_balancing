from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from pydy.codegen.code import generate_ode_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()

#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
one_frame = ReferenceFrame('1')
two_frame = ReferenceFrame('2')

#Sets up symbols for joint angles
theta1, theta2 = dynamicsymbols('theta1, theta2')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
one_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
two_frame.orient(one_frame, 'Axis', (theta2, one_frame.z))

#Sets up points for the joints and places them relative to each other
one = Point('1')
one_length = symbols('l_1')
two = Point('2')
two.set_pos(one, one_length*one_frame.y)
three = Point('3')
two_length_x = symbols('l_2x')
two_length_y = symbols('l_2y')
three.set_pos(two, two_length_x*inertial_frame.x + two_length_y*inertial_frame.y)

#Sets up the angular velocities
omega1, omega2 = dynamicsymbols('omega1, omega2')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega1 - theta1.diff(),
                                    omega2 - theta2.diff()]

#Sets up the rotational axes of the angular velocities
one_frame.set_ang_vel(inertial_frame, omega1*inertial_frame.z)
one_frame.ang_vel_in(inertial_frame)
two_frame.set_ang_vel(one_frame, omega2*inertial_frame.z)
two_frame.ang_vel_in(inertial_frame)

#Sets up the linear velocities of the points on the linkages
one.set_vel(inertial_frame, 0)
two.v2pt_theory(one, inertial_frame, one_frame)
two.vel(inertial_frame)
three.v2pt_theory(two, inertial_frame, two_frame)
three.vel(inertial_frame)

#Sets up the masses of the linkages
one_mass, two_mass = symbols('m_1, m_2')

#Defines the linkages as particles
twoP = Particle('twoP', two, one_mass)
threeP = Particle('threeP', three, two_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
two_grav_force_vector = -1*one_mass*g*inertial_frame.y
two_grav_force = (two, two_grav_force_vector)
three_grav_force_vector = -1*two_mass*g*inertial_frame.y
three_grav_force = (three, three_grav_force_vector)

#Sets up joint torques
one_torque, two_torque = dynamicsymbols('T_1, T_2')
one_torque_vector = one_torque*inertial_frame.z - two_torque*inertial_frame.z
one_link_torque = (one_frame, one_torque_vector)

two_torque_vector = two_torque*inertial_frame.z
two_link_torque = (two_frame, two_torque_vector)

#Generalized coordinates
coordinates = [theta1, theta2]

#Generalized speeds
speeds = [omega1, omega2]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [two_grav_force,
         three_grav_force,
         one_link_torque,
         two_link_torque]
bodies = [twoP, threeP]

fr, frstar = kane.kanes_equations(loads, bodies)
frplusfrstar = simplify(trigsimp(fr + frstar))
mass_matrix = simplify(trigsimp(kane.mass_matrix_full))

forcing_vector = trigsimp(kane.forcing_full)

constants = [one_length,
             one_mass,
             two_length_x,
             two_length_y,
             two_mass,
             g]
#Specified contains the matrix for the input torques
specified = [one_torque, two_torque]

#Specifies numerical constants for inertial/mass properties
#Robot Params
numerical_constants = array([1.0,  # leg_length[m]
                             1.0, # leg_mass[kg]
			     2.0, # body_length[m]
                             2.0,  # body_mass[kg]
                             9.81]    # acceleration due to gravity [m/s^2]
                             )

#Set input torques to 0
numerical_specified = zeros(2)

parameter_dict = dict(zip(constants, numerical_constants))

ke_energy = simplify(kinetic_energy(inertial_frame, twoP, threeP))

forcing = simplify(kane.forcing)
forcing_vector = kane.forcing_full

mass_matrix = simplify(kane.mass_matrix)
mass_matrix_full = kane.mass_matrix_full

torques = Matrix(specified)

zero_omega = dict(zip(speeds, zeros(2)))

g_terms = simplify(forcing.subs(zero_omega) - torques)

g_terms_1 = g_terms[0]
g_terms_2 = g_terms[1]

coriolis = simplify(forcing - g_terms - torques)

t_1i = simplify(two.pos_from(one).express(inertial_frame).to_matrix(inertial_frame))

t_2i = simplify(three.pos_from(one).express(inertial_frame).to_matrix(inertial_frame))
time = symbols('t')

com = (t_1i*one_mass + t_2i*two_mass)/(one_mass+two_mass)

com_dot = com.diff(time)
com_ddot = com_dot.diff(time).as_mutable()
des_com_ang_acc_x, des_com_ang_acc_y = dynamicsymbols('c_x, c_y')
com_ddot[0] = com_ddot[0] - des_com_ang_acc_x
com_ddot[1] = com_ddot[1] - des_com_ang_acc_y

desired_ang_acc_from_com_acc = solve(com_ddot, [theta1.diff(time).diff(time), theta2.diff(time).diff(time)])

des_t1_acc = desired_ang_acc_from_com_acc.get(theta1.diff(time).diff(time))
des_t2_acc = desired_ang_acc_from_com_acc.get(theta2.diff(time).diff(time))
