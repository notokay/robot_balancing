from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp, lambdify
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from pydy.codegen.code import generate_ode_function
import pickle
import matplotlib.animation as animation
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, constants, numerical_specified, leg_mass, body_mass, g
from double_pendulum_com import getj_0j_1
from double_pendulum_coupling import tau_a, desired_u
from sympy import Matrix

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants,
                                        coordinates, speeds, specified)

numerical_constants = array([0.5,  # leg_length[m]
                             7.0, # leg_mass[kg]
                             0.75, # body_length[m]
                             8.0,  # body_mass[kg]
                             9.81]    # acceleration due to gravity [m/s^2]
                             )
parameter_dict = dict(zip(constants, numerical_constants))
#Initial Conditions for speeds and positions
x0 = zeros(4)
x0[0] = deg2rad(.0)
x0[1] = deg2rad(.0)

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 30.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

dt = 1./frames_per_sec

right_hand_side(x0, 0.0, args)

torque_vector = []
time_vector = []
x_vec = []
ang_vec = []
vel_vec = []
error_vec = []
idx_vector = []
time_vector = []
torque_vector = []
counter = 0
goal = False 
j_0j_1 = getj_0j_1()

j_0 = j_0j_1[0]
j_1 = j_0j_1[1]
j_0T = j_0.transpose()
j_1T = j_1.transpose()
grav = Matrix([0, g, 0])
Fg1 = grav*leg_mass
Fg2 = grav*body_mass
gt = j_0T*Fg1 + j_1T*Fg2
gt = gt.subs(parameter_dict)
forcing_matrix = simplify(kane.forcing)
speeds_zero_dict = dict(zip([omega1, omega2], [0,0]))
subbed_eq = simplify(forcing_matrix.subs(speeds_zero_dict))
num_eq = simplify(subbed_eq.subs(parameter_dict))

lam_tg1 = lambdify([theta1, theta2], gt[0])
lam_tg2 = lambdify([theta1, theta2], gt[1])

tau_a = tau_a.subs(parameter_dict)

lam_tau = lambdify([theta1, theta2, omega1, omega2, desired_u], tau_a)

def gravity_compensation(x, t):
#  returnval = [lam_tg1(x[0], x[1]), lam_tg2(x[0], x[1])]
  returnval = [0,0]
  returnval[0] = x[0] - 5*sin(t)
  returnval[1] = returnval[1] + lam_tau(x[0], x[1], x[2], x[3], returnval[0])
  returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

#def com_regulator(x, t):
  

args['specified'] = gravity_compensation

y = odeint(right_hand_side, x0, t, args=(args,))

x1 = -1*numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

x2 = x1 + -1*numerical_constants[2]*sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[2]*cos(y[:,0] + y[:,1])


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

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=dt*1000, blit=True, init_func=init)
#ani.save('acrobot_zeroc_0_0_disturbance_initial_K.mp4')
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(t, rad2deg(y[:,:2]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle[deg]')
ax1.legend(["${}$".format(vlatex(c)) for c in coordinates])
"""
plot(time_vector, tracking_vector)
#plot(time_vector, curr_vector)
xlabel('Time')
ylabel('angle')
plt.show()
"""
ax3.plot(time_vector, torque_vector)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Torques')
ax3.legend(["${}$".format(vlatex(b)) for b in specified])

ax2.plot(t, rad2deg(y[:, 2:]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular Rate [deg/s]')
ax2.legend(["${}$".format(vlatex(s)) for s in speeds])

plt.show()
"""
plot(time_vector, idx_vector)
xlabel('t')
ylabel('idx')
plt.show()
"""
