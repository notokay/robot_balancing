from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


# Orientations
# ============

theta1, theta2, theta3 = dynamicsymbols('theta1, theta2, theta3')

inertial_frame = ReferenceFrame('I')

l_leg_frame = ReferenceFrame('L')

l_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

body_frame = ReferenceFrame('B')

body_frame.orient(l_leg_frame, 'Axis', (theta2, l_leg_frame.z))

r_leg_frame = ReferenceFrame('R')

r_leg_frame.orient(body_frame, 'Axis', (theta3, body_frame.z))

# Point Locations
# ===============

# Joints
# ------

l_leg_length, hip_width = symbols('l_L, h_W')

l_ankle = Point('LA')

l_hip = Point('LH')
l_hip.set_pos(l_ankle, l_leg_length * l_leg_frame.y)

r_hip = Point('RH')
r_hip.set_pos(l_hip, hip_width * body_frame.y)

# Center of mass locations
# ------------------------

l_leg_com_length, body_com_length, r_leg_com_length, body_com_height = symbols('d_L, d_B, d_R, d_BH')

l_leg_mass_center = Point('LL_o')
l_leg_mass_center.set_pos(l_ankle, l_leg_com_length * l_leg_frame.y)

body_mass_center = Point('B_o')
body_middle = Point('B_m')
body_middle.set_pos(l_hip, body_com_length*body_frame.y)
body_mass_center.set_pos(body_middle, body_com_height*body_frame.x)

r_leg_mass_center = Point('RL_o')
r_leg_mass_center.set_pos(r_hip, -1*r_leg_com_length * r_leg_frame.y)

# Define kinematical differential equations
# =========================================

omega1, omega2, omega3 = dynamicsymbols('omega1, omega2, omega3')

time = symbols('t')

kinematical_differential_equations = [omega1 - theta1.diff(time),
                                      omega2 - theta2.diff(time),
                                      omega3 - theta3.diff(time)]

# Angular Velocities
# ==================

l_leg_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)

body_frame.set_ang_vel(l_leg_frame, omega2 * l_leg_frame.z)

r_leg_frame.set_ang_vel(body_frame, omega3 * body_frame.z)

# Linear Velocities
# =================

l_ankle.set_vel(inertial_frame, 0)

l_leg_mass_center.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

l_hip.v2pt_theory(l_ankle, inertial_frame, l_leg_frame)

body_mass_center.v2pt_theory(l_hip, inertial_frame, body_frame)

r_hip.v2pt_theory(l_hip, inertial_frame, body_frame)

r_leg_mass_center.v2pt_theory(r_hip, inertial_frame, r_leg_frame)

# Mass
# ====

l_leg_mass, body_mass, r_leg_mass = symbols('m_L, m_B, m_R')

# Inertia
# =======

l_leg_inertia, body_inertia, r_leg_inertia = symbols('I_Lz, I_Bz, I_Rz')

l_leg_inertia_dyadic = inertia(l_leg_frame, 0, 0, l_leg_inertia)

l_leg_central_inertia = (l_leg_inertia_dyadic, l_leg_mass_center)

body_inertia_dyadic = inertia(body_frame, 0, 0, body_inertia)

body_central_inertia = (body_inertia_dyadic, body_mass_center)

r_leg_inertia_dyadic = inertia(r_leg_frame, 0, 0, r_leg_inertia)

r_leg_central_inertia = (r_leg_inertia_dyadic, r_leg_mass_center)

# Rigid Bodies
# ============

l_leg = RigidBody('Lower Leg', l_leg_mass_center, l_leg_frame,
                      l_leg_mass, l_leg_central_inertia)

body = RigidBody('Upper Leg', body_mass_center, body_frame,
                      body_mass, body_central_inertia)

r_leg = RigidBody('R_Leg', r_leg_mass_center, r_leg_frame,
                  r_leg_mass, r_leg_central_inertia)

# Gravity
# =======

g = symbols('g')

l_leg_grav_force = (l_leg_mass_center,
                        -l_leg_mass * g * inertial_frame.y)
body_grav_force = (body_mass_center,
                        -body_mass * g * inertial_frame.y)
r_leg_grav_force = (r_leg_mass_center, -r_leg_mass * g * inertial_frame.y)

# Joint Torques
# =============

l_ankle_torque, l_hip_torque, r_hip_torque = dynamicsymbols('T_a, T_k, T_h')

l_leg_torque = (l_leg_frame,
                    l_ankle_torque * inertial_frame.z - l_hip_torque *
                    inertial_frame.z)

body_torque = (body_frame,
                    l_hip_torque * inertial_frame.z - r_hip_torque *
                    inertial_frame.z)

r_leg_torque = (r_leg_frame, r_hip_torque * inertial_frame.z)

# Equations of Motion
# ===================

coordinates = [theta1, theta2, theta3]

speeds = [omega1, omega2, omega3]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [l_leg_grav_force,
         body_grav_force,
         r_leg_grav_force,
         l_leg_torque,
         body_torque,
         r_leg_torque]

bodies = [l_leg, body, r_leg]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

rcParams['figure.figsize'] = (14.0, 6.0)

# List the symbolic arguments
# ===========================

# Constants
# ---------

constants = [l_leg_length,
             l_leg_com_length,
             l_leg_mass,
             l_leg_inertia,
             hip_width,
             body_com_length,
             body_mass,
             body_inertia,
             r_leg_com_length,
             r_leg_mass,
             r_leg_inertia,
             body_com_height,
             g]

# Time Varying
# ------------

coordinates = [theta1, theta2, theta3]

speeds = [omega1, omega2, omega3]

specified = [l_ankle_torque, l_hip_torque, r_hip_torque]

# Generate RHS Function
# =====================

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)

# Specify Numerical Quantities
# ============================

initial_coordinates = array([-0.0781287328725265, 0.936118326612847, -0.857989593740321])
#initial_coordinates = array([0,0,0])

#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([1.035,  # l_leg_length [m]
                             0.58,  # l_leg_com_length [m]
                             23.689,  # l_leg_mass [kg]
                             0.1,  # l_leg_inertia [kg*m^2]
                             0.4,  # hip_width [m]
                             0.2,  # body_com_length
                             32.44,  # body_mass [kg]
                             1.485,  # body_inertia [kg*m^2]
                             0.193,  # r_leg_com_length [m]
                             23.689,  # r_leg_mass [kg]
                             0.1,  # r_leg_inertia [kg*m^2]
                             0.305, #body_com_height
                             9.81],  # acceleration due to gravity [m/s^2]
                           )

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 10.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

#Create dictionaries for the values for equilibrium point
equilibrium_point = zeros(len(coordinates + speeds))
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))
parameter_dict = dict(zip(constants, numerical_constants))

#Jacobian of the forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.jacobian(specified)

#Substitute in the values for the variables in the forcing vector
F_A = F_A.subs(equilibrium_dict)
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(equilibrium_dict).subs(parameter_dict)

#Convert into a floating point numpy array
F_A = array(F_A.tolist(), dtype = float)
F_B = array(F_B.tolist(), dtype = float)

M = mass_matrix.subs(equilibrium_dict)

M = M.subs(parameter_dict)
M = array(M.tolist(), dtype = float)

#Compute the state A and input B values for linearized function
A = dot(inv(M), F_A)
B = dot(inv(M), F_B)

Q = 300*eye(6)
R = eye(3)

S = solve_continuous_are(A, B, Q, R)
K = dot(dot(inv(R), B.T), S)

torque_vector = []
time_vector = []
def controller(x, t):
  torque_vector.append([200*sin(t), -100*sin(t),50*sin(t)] )
  time_vector.append(t)
  return [200*sin(t), -100*sin(t), 50*sin(t)]
def good_controller(x, t):
  temp = -dot(K,x)
#  temp[0] = 0
  torque_vector.append(temp)
  time_vector.append(t)
  return temp

def pi_controller(x, t):
  #desired = array([0.09,0.32,0.55])
  desired = initial_coordinates
  diff = desired - x[:3]
  diff = array([1, 1, 1])*diff
  torque_vector.append(diff)
  time_vector.append(t)
  return array([0, -55,0])

args['specified'] = pi_controller

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LA_x = numerical_constants[0]*sin(y[:,0])
LA_y = numerical_constants[0]*cos(y[:,0])

RH_x = LA_x + numerical_constants[4]*sin(y[:,1])
RH_y = LA_y + numerical_constants[4]*cos(y[:,1])

RA_x = RH_x + numerical_constants[8]*2*sin(y[:,2])
RA_y = RH_y + numerical_constants[8]*2*cos(y[:,2])

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
  thisx = [0, LA_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LA_y[i], RH_y[i], RA_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%(i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('triple_pendulum_bodystable_withtorque.mp4')
plt.show()

plot(t, rad2deg(y[:,:3]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(time_vector, torque_vector)
xlabel('Time [s]')
ylabel('Angle torques')
legend(["${}$".format(vlatex(ta)) for ta in specified])
plt.show()

plot(t, rad2deg(y[:, 3:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
