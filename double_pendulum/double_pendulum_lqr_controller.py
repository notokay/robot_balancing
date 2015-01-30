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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, constants, numerical_specified

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

#Initial Conditions for speeds and positions
x0 = zeros(4)
x0[0] = deg2rad(-15.0)
x0[1] = deg2rad(15.0)

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

dt = 1./frames_per_sec

right_hand_side(x0, 0.0, args)

torque_vector = []
time_vector = []

inputK = open('double_pendulum_LQR_K_murray_params.pkl','rb')
inputa1 = open('equils_a1_useful_murray_params.pkl','rb')
inputa2 = open('equils_a2_useful_murray_params.pkl','rb')
inputtor = open('equils_torques_useful_murray_params.pkl','rb')
inputG = open('double_pendulum_gain_coefs_murray_params.pkl', 'rb')

K = pickle.load(inputK)
a1 = pickle.load(inputa1)
a1 = np.asarray(a1, dtype = float)
a2 = pickle.load(inputa2)
a2 = np.asarray(a2, dtype = float)
torques = pickle.load(inputtor)
gain_coefs = pickle.load(inputG)

inputK.close()
inputa1.close()
inputa2.close()
inputtor.close()
inputG.close()

c, gt1, gt1sq, gt1cb, gt2, gt2sq, gt2cb, go1, go1sq, go1cb, go2, go2sq, go2cb, t1, t2, o1, o2 = dynamicsymbols('c, gt1, gt1sq, gt1cb, gt2, gt2sq, gt2cb, go1, go1sq, go1cb, go2, go2sq, go2cb, t1, t2, o1, o2')

A = c + gt1*t1 + gt1sq*t1**2 + gt1cb*t1**3 + gt2*t2 + gt2sq*t2**2 + gt2cb*t2**3 + go1*o1 + go1sq*o1**2 + go1cb*o1**3 + go2*o2 + go2sq*o2**2 + go2cb*o2**3

gain_funcs = [0,0,0,0]

for element in gain_coefs:
  d = dict(zip([c, gt1, gt1sq, gt1cb, gt2, gt2sq, gt2cb, go1, go1sq, go1cb, go2, go2sq, go2cb], element))
  func = A.subs(d)
  gain_funcs.append(lambdify((t1, t2, o1, o2), func))

a1_order = np.argsort(a1)
a1 = np.array(a1)[a1_order]
a2 = np.array(a2)[a1_order]
torques = np.array(torques)[a1_order]

x_vec = []
ang_vec = []
vel_vec = []
error_vec = []
idx_vector = []
time_vector = []
torque_vector = []
counter = 0
goal = False 

def lqr_controller(x, t):
  global counter
  global gain_funcs
  global a1
  global a2
  global torques
  global goal
  K_func_gains = [[0., 0., 0., 0.], [1., 1., 1., 1.]]
  K_func_gains = asarray(K_func_gains)
  a1_error = [b*b for b in a1-x[0]]
  a2_error = [b*b for b in a2-x[1]]
  tot_error = [a+b for a, b in zip(a1_error, a2_error)]
  tot_error = np.array(tot_error)
  min_errors = np.where(tot_error == tot_error.min())
  if(goal):
    closest_equil = 1134
  else:
    closest_equil = min_errors[0][0]
  if(x[2] < 0.001 and x[2] > -0.001 and x[3] < 0.001 and x[3] > -0.001):
    goal = True
  print(closest_equil)
  x0 = x[0] - a1[closest_equil]
  x1 = x[1] - a2[closest_equil]
  error_vec.append([x0, x1])
  idx_vector.append(closest_equil)
  time_vector.append(t)
  x_vec.append(x)
  vel_vec.append([x[2], x[3]])
  ang_vec.append([x[0], x[1]])
  if(counter == 0):
    for i in np.arange(len(K_func_gains)):
      for j in np.arange(len(K_func_gains[i])):
        if(K_func_gains[i][j] != 0.0):
          K_func_gains[i][j] = gain_funcs[i*4+j](x0, x1, x[2], x[3])
    counter = counter + 1
  else:
    for i in np.arange(len(K_func_gains)):
      for j in np.arange(len(K_func_gains[i])):
        if(K_func_gains[i][j] != 0.0):
          K_func_gains[i][j] = gain_funcs[i*4+j](x0, x1, x[2], x[3])
  returnval = dot(K_func_gains, [x0, x1, x[2], x[3]])
  returnval[0] = 0.0
  returnval[1] = torques[closest_equil] - returnval[1]
  counter = counter + 1
  torque_vector.append(returnval)
  return returnval

args['specified'] = lqr_controller

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

plot(time_vector, torque_vector)
xlabel('Time [s]')
ylabel('Angle 1 torque')
plt.show()
"""
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
