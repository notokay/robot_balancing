from numpy import sin, cos, cumsum, dot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify
from scipy.integrate import odeint
import numpy as np

from single_pendulum_setup import *

# Specify Numerical Quantities
# ============================

initial_coordinates = [0.1]

initial_speeds = zeros(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([15.0,
                             3.0,
                             9.81],  # acceleration due to gravity [m/s^2]
                           )
parameter_dict = dict(zip(constants, numerical_constants))
args = {'constants': numerical_constants,
        'specified': array([0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

x1 = -1*numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i]]
    thisy = [0, y1[i]]

    line.set_data(thisx, thisy)
    time_text.set_text('time = %.1f' % (i*dt))
    return line, time_text

# choose the interval based on dt and the time to animate one step
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()

