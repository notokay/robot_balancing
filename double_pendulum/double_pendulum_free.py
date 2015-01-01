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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants
#from utils import controllable

init_vprinting()


rcParams['figure.figsize'] = (14.0, 6.0)

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
numerical_specified = array([0,0])

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
