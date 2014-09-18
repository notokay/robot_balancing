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
trigsimp(fr + frstar)
mass_matrix = trigsimp(kane.mass_matrix_full)

forcing_vector = trigsimp(kane.forcing_full)

rcParams['figure.figsize'] = (14.0, 6.0)

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

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants,
                                        coordinates, speeds, specified)

#Initial Conditions for speeds and positions
x0 = zeros(4)
x0[:2] = deg2rad(40.0)
x0[1] = deg2rad(120)
#Specifies numerical constants for inertial/mass properties
#numerical_constants = array([1.035,  # leg_length[m]
#                             0.58,   # leg_com_length[m]
#                             23.779, # leg_mass[kg]
#                             0.383,  # leg_inertia [kg*m^2]
#                             0.305,  # body_com_length [m]
#                             32.44,  # body_mass[kg]
#                             1.485,  # body_inertia [kg*m^2]
#                             9.81],    # acceleration due to gravity [m/s^2]
#                             )

numerical_constants = array([1.0,  # leg_length[m]
                             0.5,   # leg_com_length[m]
                             5.0, # leg_mass[kg]
                             1.0,  # leg_inertia [kg*m^2]
                             0.5,  # body_com_length [m]
                             5,  # body_mass[kg]
                             1.0,  # body_inertia [kg*m^2]
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
#ani.save('double_pendulum_free.mp4')
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
