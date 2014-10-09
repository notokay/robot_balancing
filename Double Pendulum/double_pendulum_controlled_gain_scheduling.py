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
from double_pendulum_setup import theta1, theta2, ankle, leg_length, waist, omega1, omega2, ankle_torque, waist_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants,
                                        coordinates, speeds, specified)

#Initial Conditions for speeds and positions
x0 = zeros(4)
#x0[0] = 
#x0[1] = 0.2
x0[0] = 0.0
x0[1] = 0.0
#Specifies numerical constants for inertial/mass properties
numerical_constants = array([1.035,  # leg_length[m]
                             0.58,   # leg_com_length[m]
                             23.779, # leg_mass[kg]
                             0.383,  # leg_inertia [kg*m^2]
                             0.305,  # body_com_length [m]
                             32.44,  # body_mass[kg]
                             1.485,  # body_inertia [kg*m^2]
                             9.81],    # acceleration due to gravity [m/s^2]
                             )
#Set input torques to 0
numerical_specified = [0,0]

args = {'constants': numerical_constants,
        'specified': numerical_specified}

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time*frames_per_sec)

right_hand_side(x0, 0.0, args)

#Create dictionaries for the values for the equilibrium point of (0,0) i.e. pointing straight up
#equilibrium_point = zeros(len(coordinates + speeds))
equilibrium_point = [0.1383,-0.859099999,0,0]
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))
parameter_dict = dict(zip(constants, numerical_constants))

tor_dict = dict(zip([ankle_torque], [0]))

#Jacobian of the forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.subs(tor_dict).jacobian(specified)
#F_B = forcing_vector.jacobian(specified)

#Substitute in the values for the variables in the forcing vector
F_A = F_A.subs(equilibrium_dict)
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(equilibrium_dict).subs(parameter_dict)

#Convert into a floating point numpy array
F_A = array(F_A.tolist(), dtype=float)
F_B = array(F_B.tolist(), dtype=float)

M = mass_matrix.subs(equilibrium_dict)

M = M.subs(parameter_dict)
M = array(M.tolist(), dtype = float)

#Compute the state A and input B values for our linearized function
A = dot(inv(M), F_A)
B = dot(inv(M), F_B)

#Makes sure our function is controllable
#assert controllable(A,B)

Q = ((1/0.6)**2)*eye(4)
R = ((1/50.0)**2)*eye(2)

S = solve_continuous_are(A, B, Q, R)
gainK = dot(dot(inv(R), B.T), S)

torque_vector = []
time_vector = []

inputK = open('double_pen_LQR_K_zoom.pkl','rb')
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

def test_controller(x,t):
  global lasttime
  global lastk
  global counter
  global lastidx
  torquelim = 200
  if(counter == 0):
    lastidx = np.abs(angle_1-x[0]).argmin()
    counter = counter + 1
    print("first round")
    idx = lastidx
  else:
    idx = lastidx
  returnval = -dot(K[idx],x)
  tracking_vector.append(t)
  returnval[0] = 0
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(t > 0.5 and t < 2.00):
    returnval[0] = 4
    returnval[1] = returnval[1] - 4
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
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

args['specified'] = test_controller

y = odeint(right_hand_side, x0, t, args=(args,))

x1 = numerical_constants[0]*sin(y[:,0])
y1 = numerical_constants[0]*cos(y[:,0])

x2 = x1 + numerical_constants[4]*2*sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[4]*2*cos(y[:,0] + y[:,1])

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

plot(t, rad2deg(y[:,:2]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(time_vector, tracking_vector)
#plot(time_vector, curr_vector)
xlabel('Time')
ylabel('angle')
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

plot(time_vector, idx_vector)
xlabel('t')
ylabel('idx')
plt.show()
