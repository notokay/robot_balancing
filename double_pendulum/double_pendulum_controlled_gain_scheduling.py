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
import pickle
import matplotlib.animation as animation
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants, numerical_specified

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants,
                                        coordinates, speeds, specified)

#Initial Conditions for speeds and positions
x0 = zeros(4)
x0[0] = deg2rad(2.0)
x0[1] = deg2rad(-2.0)

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

right_hand_side(x0, 0.0, args)

torque_vector = []
time_vector = []

inputK = open('double_pen_LQR_K_robot.pkl','rb')
inputa1 = open('double_pen_angle_1_zoom.pkl','rb')
inputa2 = open('double_pen_angle_2_zoom.pkl','rb')
inputtor = open('double_pen_equil_torques_zoom.pkl','rb')

K = pickle.load(inputK)
angle_1 = pickle.load(inputa1)
angle_1 = np.asarray(angle_1, dtype = float)
angle_2 = pickle.load(inputa2)
torques = pickle.load(inputtor)

inputK.close()
inputa1.close()
inputa2.close()
inputtor.close()
lasttime = 0.0
lastk = []
lasttor = 0.0
idx_vector = []
lastidx = 0
counter = 0
tracking_vector = []
curr_vector = []

def stay_controller(x,t):
  global lastidx
  global counter
  torquelim = 200
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    counter = counter + 1
    print("first round")
    print(lastidx)
  if(x[3] < 1  and x[3] > -1):
#    idx = (np.abs(angle_1 - x[0])).argmin()
#    lastidx = idx
    idx = lastidx
    idx_vector.append(lastidx)
    print(idx)
  else:
    idx = (np.abs(angle_1 - x[0])).argmin()
    lastidx = idx
    idx_vector.append(idx)
    print(idx)
  returnval = -dot(K[idx], x)
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(x[0] > 0.22):
    #returnval[0] = 1000*(0.21 - x[0])
    #if(returnval[0] < -100):
    #  returnval[0] = -100
    returnval[1] = 0
  if(x[0] < -0.22):
    #returnval[0] = -1000*(x[0] + 0.21) 
    #if(returnval[0] > 100):
    #  returnval[0] = 100
    returnval[1] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  if(t < 1.25 and t > 1):
    returnval[0] = -40
  tracking_vector.append([angle_1[idx], angle_2[idx]])
  curr_vector.append([x[0], x[1]])
  return returnval

def zero_controller(x,t):
  global lastidx
  global counter
  torquelim = 200
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    counter = counter + 1
    idx = lastidx
    returnval = -dot(K[lastidx],x)
    print("first round")
    print(lastidx)
  if(x[2] < 0.2  and x[2] > -0.2):
    idx = (np.abs(angle_1 - x[0])).argmin()
    if((idx + 5) > 91):
      idx = idx - 4
    if((idx - 5) < 91):
      idx = idx + 4
    lastidx = idx
    returnval = -dot(K[idx], x)
    idx_vector.append(lastidx)
    print(idx)
  else:
    idx = lastidx
    idx_vector.append(lastidx)
    returnval = -dot(K[idx], x)
  tracking_vector.append([angle_1[idx], angle_2[idx]])
  curr_vector.append([x[0], x[1]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(x[0] > 0.22):
    #returnval[0] = 1000*(0.21 - x[0])
    #if(returnval[0] < -100):
    #  returnval[0] = -100
    returnval[1] = 0
  if(x[0] < -0.22):
    #returnval[0] = -1000*(x[0] + 0.21) 
    #if(returnval[0] > 100):
    #  returnval[0] = 100
    returnval[1] = 0
  if(t > 0.5 and t < 2.00):
    returnval[0] = 4
    returnval[1] = returnval[1] - 4
#  if(t > 1.0 and t < 1.15):
#    returnval[0] = 10
#  if(t > 1.3 and t < 1.45):
#    returnval[0] = 10
#  if(t > 1.7 and t < 1.85):
#    returnval[0] = 10

  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def path_controller(x,t):
  global lastidx
  global counter
  torquelim = 200
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    counter = counter + 1
    returnval = -dot(K[lastidx], x)
    print("first round")
    print(lastidx)
  if(x[2] < 0.5  and x[2] > -.5):
    if(lastidx > 0 and counter%50 ==0):
      lastidx = lastidx - 2
      print("i am slow")
    print(lastidx)
    returnval = -dot(K[lastidx], x)
    idx_vector.append(lastidx)
  else:
    idx = (np.abs(angle_1 - x[0])).argmin()
    lastidx = idx
    idx_vector.append(lastidx)
    returnval = -dot(K[idx], x)
    print(idx)
  tracking_vector.append([angle_1[lastidx], angle_2[lastidx]])
  counter = counter + 1
  curr_vector.append([x[0], x[1]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(x[0] > 0.22):
    #returnval[0] = 1000*(0.21 - x[0])
    #if(returnval[0] < -100):
    #  returnval[0] = -100
    returnval[1] = 0
  if(x[0] < -0.22):
    #returnval[0] = -1000*(x[0] + 0.21) 
    #if(returnval[0] > 100):
    #  returnval[0] = 100
    returnval[1] = 0
    
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval


def lqr_controller(x,t):
  global lastk
  if(t==0):
    idx = np.abs(angle_1 - x[0]).argmin()
    print(idx)
    lastk = K[idx]
  returnval = -dot(lastk,x)
  #if(x[0] > 0.21):
  #  returnval[0] = 1000*(0.21 - x[0])
  #if(x[0] < 0.21):
  #  returnval[0] = -1000*(x[0]+0.21) 
  returnval[0] = 500*(x0[0]-x[0])
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
  
def pid_controller(x,t):
  diff = [x0[0] - x[0], x0[1] - x[1]]
  diff[1] = diff[1]+diff[0]
  diff[0]=0
  torque_vector.append(diff)
  time_vector.append(t)
  return -100*diff

def local_controller(x,t):
  idx = np.abs(angle_1 - x[0]).argmin()
  gainK = K[idx]
  returnval = -dot(gainK,x)
  if(returnval[1] > 300):
    returnval[1] = 300
  if(returnval[1] < -300):
    returnval[1] = -300
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
def trim_controller(x,t):
  idx = np.abs(angle_1 - x[0]).argmin()
  return [0, torques[idx]]

#args['specified'] = test_controller

y = odeint(right_hand_side, x0, t, args=(args,))

x1 = numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

x2 = x1 + numerical_constants[4]*sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[4]*cos(y[:,0] + y[:,1])

p_energy_vector = []
k_energy_vector = []
tot_ke = []
tot_pe = []

for i in y[:,:2]:
  coord_dict = dict(zip(coordinates, i))
  p_energy = (leg.potential_energy.subs(coord_dict).subs(parameter_dict), body.potential_energy.subs(coord_dict).subs(parameter_dict))
  p_energy_vector.append(p_energy)
  tot_pe.append(p_energy[0] + p_energy[1])

for p,s in zip(y[:,:2], y[:,2:]):
  speeds_dict = dict(zip(speeds,s))
  coords_dict = dict(zip(coordinates, p))
  tot_ke.append(ke_body.subs(speeds_dict).subs(coords_dict).subs(parameter_dict))

tot_e = []
for i, j in zip(tot_ke, tot_pe):
  tot_e.append(i+j)

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
ani.save('acrobot_zeroc_0_0_disturbance_initial_K.mp4')
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(3)

ke1, ke2 = dynamicsymbols('ke1, ke2')
pe1, pe2 = dynamicsymbols('pe1, pe2')
ke, pe,tot = dynamicsymbols('ke, pe, tot')

energies = [pe1,pe2,ke1,ke2]
energy = [pe, ke, tot]




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

ax3.plot(t, tot_pe)
ax3.plot(t, tot_ke)
ax3.plot(t, tot_e)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Energy')
ax3.legend(["${}$".format(vlatex(e)) for e in energy])
plt.show()
"""
plot(time_vector, idx_vector)
xlabel('t')
ylabel('idx')
plt.show()
"""
