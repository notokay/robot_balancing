# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from pydy.codegen.code import generate_ode_function
import matplotlib.animation as animation
#from utils import controllable

init_vprinting()
#Sets up inertial frame as well as frames for each linkage
inertial_frame = ReferenceFrame('I')
r_leg_frame = ReferenceFrame('R')
body_frame = ReferenceFrame('B')
l_leg_frame = ReferenceFrame('L')

#Sets up symbols for joint angles
theta1, theta2, theta3 = dynamicsymbols('theta1, theta2, theta3')

#Orients the leg frame to the inertial frame by angle theta1
#and the body frame to to the leg frame by angle theta2
l_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
body_frame.orient(l_leg_frame, 'Axis', (theta2, l_leg_frame.z))
r_leg_frame.orient(body_frame, 'Axis', (theta3, body_frame.z))

#Sets up points for the joints and places them relative to each other
l_ankle = Point('LA')
l_leg_length = symbols('l_L')
l_hip = Point('LH')
l_hip.set_pos(r_ankle, r_leg_length*r_leg_frame.y)
hip_width = symbols('h_W')
r_hip = Point('RH')
r_hip.set_pos(l_hip, hip_width*hip_frame.x)
r_ankle = Point('RA')
r_leg_length = symbols('r_L')
r_ankle.set_pos(r_hip, -1*r_leg_lenth*hip_frame.y)


#Sets up the centers of mass of each of the linkages
r_leg_com_length, body_com_length, l_leg_com_length = symbols('d_RL, d_B, d_LL')
l_leg_mass_center = Point('LL_o')
l_leg_mass_center.set_pos(l_ankle, l_leg_com_length*l_leg_frame.y)
body_mass_center = Point('B_o')
body_mass_center.set_pos(l_hip, body_com_length*hip_frame.x)
r_leg_mass_center = Point('RL_o')
r_leg_mass_center.set_pos(r_hip, -1*l_leg_com_length*r_leg_frame.y)

#Sets up the angular velocities
omega1, omega2, omega3 = dynamicsymbols('omega1, omega2, omega3')
#Relates angular velocity values to the angular positions theta1 and theta2
kinematic_differential_equations = [omega1 - theta1.diff(),
                                    omega2 - theta2.diff(),
                                    omega3 - theta3.diff()]

#Sets up the rotational axes of the angular velocities
l_leg_frame.set_ang_vel(inertial_frame, omega1*inertial_frame.z)
l_leg_frame.ang_vel_in(inertial_frame)
body_frame.set_ang_vel(l_leg_frame, omega2*inertial_frame.z)
body_frame.ang_vel_in(inertial_frame)
r_leg_frame.set_ang_vel(body_frame, omega3*inertial_frame.z)
r_leg_frame.ang_vel_in(inertial-frame)

#Sets up the linear velocities of the points on the linkages
l_ankle.set_vel(inertial_frame, 0)
l_leg_mass_center.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)
l_leg_mass_center.vel(inertial_frame)
l_hip.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)
l_hip.vel(inertial_frame)
body_mass_center.v2pt_theory(l_hip, inertial_frame, body_frame)
body_mass_center.vel(inertial_frame)
r_hip.v2pt_theory(l_hip, inertial_frame, body_frame)
r_hip.vel(inertial_frame)
r_leg_mass_center.v2pt_theory(r_hip, inertial_frame, r_leg_frame)
r_leg_mass_center.vel(inertial_frame)

#Sets up the masses of the linkages
l_leg_mass, body_mass, r_leg_mass = symbols('m_LL, m_B, m_RL')

#Sets up the rotational inertia of the linkages
l_leg_inertia, body_inertia, r_leg_inertia = symbols('I_LLz, I_Bz, I_RLz')

#Sets up inertia dyadics
l_leg_inertia_dyadic = inertia(l_leg_frame, 0, 0, l_leg_inertia)
l_leg_central_inertia = (l_leg_inertia_dyadic, l_leg_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)
body_central_inertia = (body_inertia_dyadic, body_mass_center)

r_leg_inertia_dyadic = inertia(r_leg_frame, 0, 0, r_leg_inertia)
r_leg_central_inertia = (r_leg_inertia_dyadic, r_leg_mass_center)

#Defines the linkages as rigid bodies
l_leg = RigidBody('LLeg', l_leg_mass_center, l_leg_frame, l_leg_mass, l_leg_central_inertia)
body = RigidBody('Body', body_mass_center, body_frame, body_mass, body_central_inertia)
r_leg = RigidBody('RLeg', r_leg_mass_center, r_leg_frame, r_leg_mass, r_leg_central_inertia)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')
l_leg_grav_force_vector = -l_leg_mass*g*inertial_frame.y
l_leg_grav_force = (l_leg_mass_center, l_leg_grav_force_vector)
body_grav_force_vector = -body_mass*g*inertial_frame.y
body_grav_force = (body_mass_center,body_grav_force_vector)
r_leg_grav_force_vector = -r_leg_mass*g*inertial_frame.y
r_leg_grav_force = (r_leg_mass_center, r_leg_)

#Sets up joint torques
l_ankle_torque, l_hip_torque, r_hip_torque = dynamicsymbols('T_la, T_lh, T_rh')
l_ankle_torque_vector = l_ankle_torque*inertial_frame.z - l_hip_torque*inertial_frame.z
l_ankle_torque = (l_leg_frame, l_ankle_torque_vector)

l_hip_torque_vector = l_hip_torque*inertial_frame.z - r_hip_torque_*inertial_frame.z
l_hip_torque = (body_frame, l_hip_torque_vector)

r_hip_torque_vector = r_hip_torque*inertial_frame.z
r_hip_torque = (r_leg_frame, r_hip_torque_vector)

#Generalized coordinates
coordinates = [theta1, theta2, theta3]

#Generalized speeds
speeds = [omega1, omega2, omega3]

#Create a KanesMethod object
kane = KanesMethod(inertial_frame, coordinates, speeds, kinematic_differential_equations)

loads = [l_leg_grav_force,
         body_grav_force,
         r_leg_grav_force,
         l_ankle_torque,
         l_hip_torque,
         r_hip_torque]
bodies = [l_leg, body, r_leg]

fr, frstar = kane.kanes_equations(loads, bodies)
trigsimp(fr + frstar)
mass_matrix = trigsimp(kane.mass_matrix_full)

forcing_vector = trigsimp(kane.forcing_full)

rcParams['figure.figsize'] = (14.0, 6.0)

constants = [l_leg_length,
             l_leg_com_length,
             l_leg_mass,
             l_leg_inertia,
             body_length
             body_com_length,
             body_mass,
             body_inertia,
             r_leg_length,
             r_leg_com_length,
             r_leg_mass,
             r_leg_inertia
             g]

#Specified contains the matrix for the input torques
specified = [l_ankle_torque, l_hip_torque, r_hip_torque]

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants,
                                        coordinates, speeds, specified)

#Initial Conditions for speeds and positions
x0 = zeros(6)
#x0[:3] = deg2rad(40.0)

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

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

right_hand_side(x0, 0.0, args)

#Create dictionaries for the values for the equilibrium point of (0,0) i.e. pointing straight up
equilibrium_point = zeros(len(coordinates + speeds))
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))
parameter_dict = dict(zip(constants, numerical_constants))

#Jacobian of the forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.jacobian(specified)

#Substitute in the values for the variables in the forcing vector
F_A = simplify(F_A.subs(equilibrium_dict))
F_A = F_A.subs(parameter_dict)
F_B = simplify(F_B.subs(equilibrium_dict).subs(parameter_dict))

#Convert into a floating point numpy array
F_A = array(F_A.tolist(), dtype=float)
F_B = array(F_B.tolist(), dtype=float)

M = mass_matrix.subs(equilibrium_dict)
simplify(M)

M = M.subs(parameter_dict)
M = array(M.tolist(), dtype = float)

#Compute the state A and input B values for our linearized function
A = dot(inv(M), F_A)
B = dot(inv(M), F_B)

#Makes sure our function is controllable
#assert controllable(A,B)

Q = eye(4)
R = eye(2)

S = solve_continuous_are(A, B, Q, R)
K = dot(dot(inv(R), B.T), S)

def controller(x,t):
  return -dot(K, x)

args['specified'] = controller

y = odeint(right_hand_side, x0, t, args=(args,))

x1 = numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

x2 = x1 + numerical_constants[4]*2*sin(y[:,1])
y2 = y1 + numerical_constants[4]*2*cos(y[:,1])

dt = 0.05

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,aspect='equal', xlim = (-2, 2), ylim = (-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time=%.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
  line.set_data([],[])
  time_text.set_text('')
  return line, time_text

def animate(i):
  thisx = [0, x1[i], x2[i]]
  thisy = [0, y1[i], y2[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%(i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
ani.save('double_pendulum.mp4')
plt.show()

plot(t, rad2deg(y[:,:2]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(t, rad2deg(y[:, 2:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
