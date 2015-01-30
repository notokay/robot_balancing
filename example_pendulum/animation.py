from pydy.viz.shapes import Cylinder, Sphere
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene
from numpy import sin, cos, cumsum, dot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from sympy import lambdify
import numpy as np

from pendulum import *

# Specify Numerical Quantities
# ============================

initial_coordinates = [deg2rad(0.), deg2rad(160.)]

initial_speeds = deg2rad(0.) * ones(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([1.0,  # lower_leg_length [m]
                             0.5,  # lower_leg_com_length [m]
                             1.0,  # lower_leg_mass [kg]
                             1.0,  # lower_leg_inertia [kg*m^2]
                             1.0,  # upper_leg_length [m]
                             0.5,  # upper_leg_com_length
                             1.0,  # upper_leg_mass [kg]
                             1.0,  # upper_leg_inertia [kg*m^2]
                             9.81],  # acceleration due to gravity [m/s^2]
                           )
parameter_dict = dict(zip(constants, numerical_constants))
args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./60

x1 = -1*numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

x2 = x1 + -1*numerical_constants[4]*sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[4]*cos(y[:,0] + y[:,1])

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform = ax.transAxes)

ke = kinetic_energy(inertial_frame, lower_leg, upper_leg)
lam_ke = lambdify([theta1, theta2, omega1, omega2], ke.subs(parameter_dict))

def energy(i):
  global y, x1, y1, x2, y2
  m1 = numerical_constants[2]
  m2 = numerical_constants[6]
  l1 = numerical_constants[0]
  l2 = numerical_constants[4]
  x0_local = y[i][0]
  x1_local = y[i][2]
  x2_local = y[i][1]
  x3_local = y[i][3]
  xe = [x1[i], x2[i]]
  ye = [y1[i], y2[i]]
  vx = np.cumsum([l1 * x1_local*cos(x0_local), l2*x3_local*cos(x2_local)])
  vy = np.cumsum([l1 * x1_local*sin(x0_local), l2*x3_local*sin(x2_local)])
  u = 9.81*(m1*ye[0] + m2*ye[1])
  k = lam_ke(y[i][0], y[i][1], y[i][2], y[i][3])
  return k + u

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text('time = %.1f' % (i*dt))
    energy_text.set_text('energy = %.6f J' % energy(i))
    return line, time_text, energy_text


# choose the interval based on dt and the time to animate one step
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()

