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
M = simplify(M)
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

torque_vector = []
time_vector = []

def controller(x,t):
  torque_vector.append([500*sin(t), -500*sin(t)])
  time_vector.append(t)
  return [500*sin(t),-500*sin(t) ]

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
ani.save('double_pendulum_controlled_spinny.mp4')
plt.show()

plot(t, rad2deg(y[:,:2]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(time_vector, torque_vector)
xlabel('Time [s]')
ylabel('Angle 1 torque')
plt.show()

plot(t, rad2deg(y[:, 2:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
