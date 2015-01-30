from numpy import sin, cos, cumsum, dot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify
from scipy.integrate import odeint
import numpy as np
import single_pendulum_setup as sp

from double_pendulum_particle_setup import *

# Specify Numerical Quantities
# ============================
right_hand_side = generate_ode_function(mass_matrix_full, forcing_vector, 
                                        constants, coordinates,
                                        speeds, specified)
initial_coordinates = [0.5, 0.5]

initial_speeds = zeros(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([0.75,
                             7.0,
                             0.5,
                             8.0,
                             9.81],  # acceleration due to gravity [m/s^2]
                           )
parameter_dict = dict(zip(constants, numerical_constants))


args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0])}

right_hand_side

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

right_hand_side(x0, 0.0, args)

com_acc = sp.kane.mass_matrix.inv()*sp.kane.forcing
sp_constants_dict = dict(zip([sb.a_mass, sb.g], [one_mass + two_mass, 

com_angle = atan(com[0]/com[1]).subs(parameter_dict)
com_length = com.norm().subs(parameter_dict)
lam_com_angle = lambdify([theta1, theta2], com_angle)
lam_com_length = lambdify([theta1, theta2], com_length)


def controller(x, t):
    
    return

args['specified'] = controller

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

x1 = -1*numerical_constants[0]*np.sin(y[:,0])
y1 = numerical_constants[0]*np.cos(y[:,0])

x2 = x1 + -1*numerical_constants[2]*np.sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[2]*np.cos(y[:,0] + y[:,1])

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
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text('time = %.1f' % (i*dt))
    return line, time_text

# choose the interval based on dt and the time to animate one step
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()

