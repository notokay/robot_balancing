from numpy import array, zeros
from sympy import symbols, simplify, trigsimp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod

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

#Sets up the centers of mass of each of the linkages
leg_com_length, body_com_length = symbols('d_L, d_B')
leg_mass_center = Point('L_o')
leg_mass_center.set_pos(ankle, leg_com_length*leg_frame.y)
body_mass_center = Point('B_o')
body_mass_center.set_pos(waist, body_com_length*body_frame.y)

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
leg_mass_center.v2pt_theory(ankle, inertial_frame, leg_frame)
leg_mass_center.vel(inertial_frame)
waist.v2pt_theory(ankle, inertial_frame, leg_frame)
waist.vel(inertial_frame)
body_mass_center.v2pt_theory(waist, inertial_frame, body_frame)
body_mass_center.vel(inertial_frame)

#Sets up the masses of the linkages
leg_mass, body_mass = symbols('m_L, m_B')

#Sets up the rotational inertia of the linkages
leg_inertia, body_inertia = symbols('I_Lz, I_Bz')

#Sets up inertia dyadics
leg_inertia_dyadic = inertia(leg_frame, 0, 0, leg_inertia)
leg_central_inertia = (leg_inertia_dyadic, leg_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)
body_central_inertia = (body_inertia_dyadic, body_mass_center)

#Defines the linkages as rigid bodies
leg = RigidBody('Leg', leg_mass_center, leg_frame, leg_mass, leg_central_inertia)
body = RigidBody('Body', body_mass_center, body_frame, body_mass, body_central_inertia)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
leg_grav_force_vector = -leg_mass*g*inertial_frame.y
leg_grav_force = (leg_mass_center, leg_grav_force_vector)
body_grav_force_vector = -body_mass*g*inertial_frame.y
body_grav_force = (body_mass_center,body_grav_force_vector)

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
bodies = [leg, body]

fr, frstar = kane.kanes_equations(loads, bodies)
frplusfrstar = simplify(trigsimp(fr + frstar))
mass_matrix = simplify(trigsimp(kane.mass_matrix_full))

forcing_vector = trigsimp(kane.forcing_full)

constants = [leg_length,
             leg_com_length,
             leg_mass,
             leg_inertia,
             body_com_length,
             body_mass,
             body_inertia,
             g]
#Specified contains the matrix for the input torques
specified = [ankle_torque, waist_torque]

#Specifies numerical constants for inertial/mass properties
numerical_constants = array([1.035,  # leg_length[m]
                             0.58,   # leg_com_length[m]
                             23.779, # leg_mass[kg]
                             0.383,  # leg_inertia [kg*m^2]
                             0.305,  # body_com_length [m]
                             32.44,  # body_mass[kg]
                             1.485,  # body_inertia [kg*m^2]
                             9.81],    # acceleration due to gravity [m/s^2]
                             )
#Set input torques to 0
numerical_specified = zeros(2)

parameter_dict = dict(zip(constants, numerical_constants))
