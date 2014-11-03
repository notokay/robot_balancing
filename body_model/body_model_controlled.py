from sympy import symbols, simplify
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
from body_model_setup import theta1, theta2, theta3,theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque, waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()
import pickle


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0.1,0.,0.,-0.])
#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

# Control
# =======

inputK = open('bm_LQR_K_useful.pkl','rb')
inputa1 = open('bm_angle_one_useful.pkl','rb')
inputa2 = open('bm_angle_two_useful.pkl','rb')
inputa3 = open('bm_angle_three_useful.pkl','rb')
inputa4 = open('bm_angle_four_useful.pkl','rb')

K = pickle.load(inputK)
a1 = pickle.load(inputa1)
a1 = np.asarray(a1, dtype = float)
a2 = pickle.load(inputa2)
a2 = np.asarray(a2, dtype = float)
a3 = pickle.load(inputa3)
a3 = np.asarray(a3, dtype = float)
a4 = pickle.load(inputa4)
a4 = np.asarray(a4, dtype = float)

inputK.close()
inputa1.close()
inputa2.close()
inputa3.close()
inputa4.close()

torque_vector = []
lastk = []
idx_vector = []
lastidx = 0
counter = 0
tracking_vector = []
curr_vector = []
time_vector = []
output_vector = []
diff_vector = []
limits_vector = []
passivity_vector = []

def calc_com(t1, t2, t3, t4):
  return t1

def controller(x, t):
  return x
def limits_only(x,t):
  returnval = [0,0,0,0]
  if(x[0] > 0.527):
    returnval[0] = -5000*(x[0] - 0.527)
  if(x[0] < -0.527):
    returnval[0] = -5000*(x[0] - 0.527) 
  if(x[1] > 0.527):
    returnval[1] = -5000*(x[1] - 0.526)
  if(x[1] < -0.527):
    returnval[1] = -5000*(x[1] - 0.527)
  if(x[2] > 0.527):
    returnval[2] = -5000*(x[2] - 0.526)
  if(x[2] < -0.527):
    returnval[2] = -5000*(x[2] - 0.527)
  if(x[3] > 0.527):
    returnval[3] = -5000*(x[3] - 0.527)
  if(x[3] < -0.527):
    returnval[3] = -5000*(x[3] - 0.527)
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def adapt_controller(x,t):
  global lastidx
  global counter
  torquelim = 500
  limit_torque = 200
  if(counter==0):
    lastidx = np.abs(a1 - x[0]).argmin()
    lastidx = lastidx + np.abs(a2[lastidx:lastidx+40] - x[1]).argmin()
    lastidx = lastidx + np.abs(a4[lastidx:lastidx+40] - x[3]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print (lastidx)
  if(x[4] < 1.0 and x[3] > -1.0):
    idx = np.abs(a1 - x[0]).argmin()
    idx = idx + np.abs(a2[idx:idx+40] - x[1]).argmin()
    idx = idx + np.abs(a4[idx:idx+40] - x[3]).argmin()
    lastidx = idx
    idx_vector.append(idx)
    print('adapt')
    print(idx)
  else:
    if(x[6] > 1.0 or x[6] < -1.0):
      idx = lastidx
      print(idx)
      idx_vector.append(lastidx)
    else:
      idx = np.abs(a1-x[0]).argmin()
      idx = idx + np.abs(a2[idx:idx+30] - x[1]).argmin()
      idx = idx + np.abs(a4[idx:idx+30] - x[3]).argmin()
      lastidx = idx
      print(idx)
      idx_vector.append(lastidx)
  if(idx > (len(K)-1)):
    idx = len(K)-1
  if(idx < 0):
    idx = 0
  returnval = -dot(K[idx], x)
  if(returnval[1] > torquelim): 
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
  if(returnval[3] < -1*torquelim):
    returnval[3] = -1*torquelim
  if(returnval[3] > torquelim):
    returnval[3] = torquelim
  if(x[0] < -1.0):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
    returnval[3] = 0
  if(x[0] > 1.0):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
    returnval[3] = 0

  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def const_controller(x,t):
  return [0,180, 180, 180]

def energy_controller(x,t):
  return x
args['specified'] = adapt_controller

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LH_x = -1*numerical_constants[0]*sin(y[:,0])
LH_y = numerical_constants[0]*cos(y[:,0])

C_x = LH_x + (numerical_constants[4]/2)*cos(y[:,1] + y[:,0])
C_y = LH_y + (numerical_constants[4]/2)*sin(y[:,1] + y[:,0])

W_x = C_x + numerical_constants[8]*cos(y[:,1] + y[:,0] + 1.57)
W_y = C_y + numerical_constants[8]*sin(y[:,1] + y[:,0] + 1.57)

B_x = W_x + numerical_constants[9]*2*cos(y[:,1] + y[:,0] + 1.57 + y[:,3])
B_y = W_y + numerical_constants[9]*2*sin(y[:,1] + y[:,0] + 1.57 + y[:,3])

RH_x = LH_x + numerical_constants[4]*cos(y[:,1] + y[:,0])
RH_y = LH_y + numerical_constants[4]*sin(y[:,1] + y[:,0])

RA_x = RH_x + numerical_constants[12]*2*sin(y[:,2] + y[:,1] + y[:,0])
RA_y = RH_y + -1*numerical_constants[12]*2*cos(y[:,2] + y[:,1] + y[:,0])

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
  thisx = [0, LH_x[i], C_x[i], W_x[i], B_x[i], W_x[i], C_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LH_y[i], C_y[i], W_y[i], B_y[i], W_y[i], C_y[i], RH_y[i], RA_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%((i*dt)/3))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('triple_pendulum_alt_stay_zero_controller.mp4')
plt.show()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

ax1.plot(t, rad2deg(y[:,:3]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle[deg]')
ax1.legend(["${}$".format(vlatex(c)) for c in coordinates])

ax3.plot(time_vector, torque_vector)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Angle torques')
ax3.legend(["${}$".format(vlatex(c)) for c in specified])
"""
plot(time_vector, idx_vector)
xlabel('time')
ylabel('idx')
plt.show()
"""
ax2.plot(t, rad2deg(y[:, 3:]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular Rate [deg/s]')
ax2.legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
