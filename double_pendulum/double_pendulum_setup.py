from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy

#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
leg_frame = ReferenceFrame('L')
body_frame = ReferenceFrame('B')

#Sets up symbols for joint angles
theta1, theta2 = dynamicsymbols('theta1, theta2')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
body_frame.orient(leg_frame, 'Axis', (theta2, leg_frame.z))

#Sets up points for the joints and places them relative to each other
ankle = Point('A')
leg_length = symbols('l_L')
waist = Point('W')
waist.set_pos(ankle, leg_length*leg_frame.y)
body = Point('B')
body_length = symbols('l_B')
body.set_pos(waist, body_length*body_frame.y)

#Sets up the angular velocities
omega1, omega2 = dynamicsymbols('omega1, omega2')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega1 - theta1.diff(),
                                    omega2 - theta2.diff()]

#Sets up the rotational axes of the angular velocities
leg_frame.set_ang_vel(inertial_frame, omega1*inertial_frame.z)
leg_frame.ang_vel_in(inertial_frame)
body_frame.set_ang_vel(leg_frame, omega2*inertial_frame.z)
body_frame.ang_vel_in(inertial_frame)

#Sets up the linear velocities of the points on the linkages
ankle.set_vel(inertial_frame, 0)
waist.v2pt_theory(ankle, inertial_frame, leg_frame)
waist.vel(inertial_frame)
body.v2pt_theory(waist, inertial_frame, body_frame)
body.vel(inertial_frame)

#Sets up the masses of the linkages
leg_mass, body_mass = symbols('m_L, m_B')

#Defines the linkages as particles
waistP = Particle('waistP', waist, leg_mass)
bodyP = Particle('bodyP', body, body_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
leg_grav_force_vector = -1*leg_mass*g*inertial_frame.y
leg_grav_force = (waist, leg_grav_force_vector)
body_grav_force_vector = -1*body_mass*g*inertial_frame.y
body_grav_force = (body,body_grav_force_vector)

#Sets up joint torques
ankle_torque, waist_torque = dynamicsymbols('T_a, T_w')
leg_torque_vector = ankle_torque*inertial_frame.z - waist_torque*inertial_frame.z
leg_torque = (leg_frame, leg_torque_vector)

body_torque_vector = waist_torque*inertial_frame.z
body_torque = (body_frame, body_torque_vector)

#Generalized coordinates
coordinates = [theta1, theta2]

#Generalized speeds
speeds = [omega1, omega2]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [leg_grav_force,
         body_grav_force,
         leg_torque,
         body_torque]
bodies = [waistP, bodyP]

fr, frstar = kane.kanes_equations(loads, bodies)
frplusfrstar = simplify(trigsimp(fr + frstar))
mass_matrix = simplify(trigsimp(kane.mass_matrix_full))

forcing_vector = trigsimp(kane.forcing_full)

constants = [leg_length,
             leg_mass,
             body_length,
             body_mass,
             g]
#Specified contains the matrix for the input torques
specified = [ankle_torque, waist_torque]

#Specifies numerical constants for inertial/mass properties
#Robot Params
#numerical_constants = array([1.035,  # leg_length[m]
#                             36.754, # leg_mass[kg]
#			     0.85, # body_length[m]
#                             91.61,  # body_mass[kg]
#                             9.81]    # acceleration due to gravity [m/s^2]
#                             )
numerical_constants = array([0.75,
                             7.0,
                             0.5,
                             8.0,
                             9.81])

#Set input torques to 0
numerical_specified = zeros(2)

parameter_dict = dict(zip(constants, numerical_constants))

ke_energy = simplify(kinetic_energy(inertial_frame, waistP, bodyP).subs(parameter_dict))

waistP.set_potential_energy(leg_mass*g*leg_length*cos(theta1))

bodyP.set_potential_energy(body_mass*g*(leg_length*cos(theta1)+body_length*cos(theta1+theta2)))

pe_energy = simplify(potential_energy(waistP, bodyP).subs(parameter_dict))
