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
from triple_pendulum_setup_alt import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
from sympy.physics.vector import init_vprinting, vlatex
from math import fabs
init_vprinting()
import pickle


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0.2,0.4,0.2])
#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

#Create dictionaries for the values for equilibrium point
equilibrium_point = zeros(len(coordinates + speeds))
#equilibrium_point[:3] = deg2rad(180.0)
equilibrium_point[0] = initial_coordinates[0]
equilibrium_point[1] = initial_coordinates[1]
equilibrium_point[2] = initial_coordinates[2]
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))

#Jacobian of the forcing vector w.r.t. states and inputs
tordict = dict(zip([l_ankle_torque], [0]))
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.subs(tordict).jacobian(specified)

#Substitute in the values for the variables in the forcing vector
F_A = F_A.subs(equilibrium_dict)
F_A = F_A.subs(parameter_dict)
F_B = F_B.subs(equilibrium_dict).subs(parameter_dict)

#Convert into a floating point numpy array
F_A = array(F_A.tolist(), dtype = float)
F_B = array(F_B.tolist(), dtype = float)

M = mass_matrix.subs(equilibrium_dict)

M = M.subs(parameter_dict)
M = array(M.tolist(), dtype = float)

#Compute the state A and input B values for linearized function
A = dot(inv(M), F_A)
B = dot(inv(M), F_B)

Q = ((1/0.6)**2)*eye(6)
R = eye(3)

S = solve_continuous_are(A, B, Q, R)
gainK = dot(dot(inv(R), B.T), S)

inputK = open('triple_pen_LQR_K_useful.pkl','rb')
inputa1 = open('triple_pen_angle_one_useful.pkl','rb')
inputa2 = open('triple_pen_angle_two_useful.pkl','rb')
inputa3 = open('triple_pen_angle_three_useful.pkl','rb')
inputt = open('triple_pendulum_trim_zoom_half.pkl','rb')

K = pickle.load(inputK)
angle_1 = pickle.load(inputa1)
a1 = angle_1
angle_1 = np.asarray(angle_1, dtype = float)
angle_2 = pickle.load(inputa2)
a2 = angle_2
angle_2 = np.asarray(angle_2, dtype = float)
angle_3 = pickle.load(inputa3)
a3 = angle_3
angle_3 = np.asarray(angle_3, dtype =float)
trim = pickle.load(inputt)
trim = np.asarray(trim, dtype = float)

inputK.close()
inputa1.close()
inputa2.close()
inputa3.close()
inputt.close()

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

def controller(x, t):
  torque_vector.append([200*sin(t), -100*sin(t),50*sin(t)] )
  time_vector.append(t)
  return [200*sin(t), -100*sin(t), 50*sin(t)]
def zero_controller(x, t):
  global lastidx
  global counter
  torquelim = 200
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    lastidx = lastidx + np.abs(angle_3[lastidx:lastidx+30] - x[2]).argmin()
    counter = counter + 1
    returnval = -dot(K[lastidx],x)
    print("first round")
    print(lastidx)
  if(x[3] < 1 and x[3] > -1):
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_3[idx:idx+30]-x[2]).argmin()
    idx_vector.append(idx)
    lastidx = idx
    print(idx)
  else:
    idx = lastidx
    idx_vector.append(idx)
  returnval = -dot(K[idx], x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
  if(x[0] < -0.46):
    returnval[1] = 0
    returnval[2] = 0
  if(x[0] > 0.46):
    returnval[1] = 0
    returnval[2] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def stay_controller(x, t):
  global lastidx
  global counter
  torquelim = 500
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    lastidx = lastidx + np.abs(angle_3[lastidx:lastidx+30] - x[2]).argmin()
    counter = counter + 1
    idx = lastidx
    print("first round")
    print(lastidx)
  if(x[3] < 1  and x[3] > -1):
    idx = lastidx
    idx_vector.append(lastidx)
  else:
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_3[idx:idx+30] - x[2]).argmin()
    print(idx)
    lastidx = idx
    idx_vector.append(lastidx)
  returnval = -dot(K[idx],x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
  returnval[0] = 0
  if(x[0] < -0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def stay_zero_controller(x,t):
  global lastidx
  global counter
  torquelim = 800
  if(counter ==0):
    lastidx = np.abs(angle_1-x[0]).argmin()
    lastidx = lastidx + np.abs(angle_2[lastidx:lastidx +30] - x[1]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print(lastidx)
  if(x[3] < 0.5 and x[3] > -0.5):
    idx = lastidx
    idx_vector.append(lastidx)
    print('stay')
    print(idx)
  elif(x[5] > 0.1 and x[5] < -0.1):
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_2[idx:idx+30] - x[1]).argmin()
    idx_vector.append(idx)
    lastidx = idx
    print('adapt')
    print(idx)
  else:
    idx = lastidx
    print(idx)
    idx_vector.append(lastidx)
  returnval = -dot(K[idx],x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
  returnval[0] = 0
  if(x[0] < -0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
def zero_stay_controller(x,t):
  global lastidx
  global counter
  torquelim = 800
  if(counter ==0):
    lastidx = np.abs(angle_1-x[0]).argmin()
    lastidx = lastidx + np.abs(angle_2[lastidx:lastidx +30] - x[1]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print(lastidx)
  if(x[3] < 1.0 and x[3] > -1.0):
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_2[idx:idx+30] - x[1]).argmin()
    idx_vector.append(lastidx)
    lastidx = idx
    print('adapt')
    print(idx)
  else:
    if(x[5] > 1.0 or x[5] < -1.0):
      idx = lastidx
      print(idx)
      idx_vector.append(lastidx)
    else:
      idx = np.abs(angle_1 - x[0]).argmin()
      idx = idx + np.abs(angle_3[idx:idx+30] - x[2]).argmin()
      lastidx = idx
      print(idx)
      idx_vector.append(lastidx)
  returnval = -dot(K[idx],x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim): 
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
 # returnval[0] = 0
  if(x[0] < -0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def limits_controller(x,t):
  global lastidx
  global counter
  torquelim = 200
  if(counter ==0):
    lastidx = np.abs(angle_1-x[0]).argmin()
    lastidx = lastidx + np.abs(angle_2[lastidx:lastidx +30] - x[1]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print(lastidx)
  if(x[3] < 1.0 and x[3] > -1.0):
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_2[idx:idx+30] - x[1]).argmin()
    lastidx = idx
    idx_vector.append(idx)
    print('adapt')
    print(idx)
  else:
    if(x[5] > 1.0 or x[5] < -1.0):
      idx = lastidx
      print(idx)
      idx_vector.append(lastidx)
    else:
      idx = np.abs(angle_1 - x[0]).argmin()
      idx = idx + np.abs(angle_3[idx:idx+30] - x[2]).argmin()
      lastidx = idx
      print(idx)
      idx_vector.append(lastidx)
  returnval = -dot(K[idx],x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim): 
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
 # returnval[0] = 0
  if(x[0] < -0.437):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.437):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[1] > 0.527):
    print(returnval[1])
    returnval[1] = -200*(x[1]-0.526)
  if(x[1] < -0.527):
    print(returnval[1])
    returnval[1] = -200*(x[1]-0.527)
  if(x[2] > 0.527):
    returnval[2] = -200*(x[2]-0.527)
  if(x[2] < -0.527):
    returnval[2] = -200*(x[2]-0.527)
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def local_controller(x, t):
  torquelim = 20
  returnval = -dot(gainK,x)
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
#  temp[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

def trim_controller(x,t):
  lastidx = np.abs(angle_1 - x[0]).argmin()
  lastidx = lastidx + np.abs(angle_3[lastidx:lastidx+30] - x[2]).argmin()
  returnval = [0, trim[lastidx][0], trim[lastidx][1]]
  idx_vector.append(lastidx)
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

args['specified'] = limits_controller

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LA_x = -1*numerical_constants[0]*sin(y[:,0])
LA_y = numerical_constants[0]*cos(y[:,0])

C_x = LA_x + numerical_constants[5]*cos(y[:,1] + y[:,0])
C_y = LA_y + numerical_constants[5]*sin(y[:,1] + y[:,0])

M_x = C_x + numerical_constants[11]*cos(y[:,1] + y[:,0] + 1.57)
M_y = C_y + numerical_constants[11]*sin(y[:,1] + y[:,0] + 1.57)

RH_x = LA_x + numerical_constants[4]*cos(y[:,1] + y[:,0])
RH_y = LA_y + numerical_constants[4]*sin(y[:,1] + y[:,0])

RA_x = RH_x + numerical_constants[8]*2*sin(y[:,2] + y[:,1] + y[:,0])
RA_y = RH_y + -1*numerical_constants[8]*2*cos(y[:,2] + y[:,1] + y[:,0])

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
  thisx = [0, LA_x[i], RH_x[i], RA_x[i], RH_x[i], C_x[i], M_x[i]]
  thisy = [0, LA_y[i], RH_y[i], RA_y[i], RH_y[i], C_y[i], M_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%(i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('triple_pendulum_alt_with_limits.mp4')
plt.show()

plot(t, rad2deg(y[:,:3]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(time_vector, torque_vector)
xlabel('Time [s]')
ylabel('Angle torques')
legend(["${}$".format(vlatex(c)) for c in specified])
plt.show()
plot(time_vector, idx_vector)
xlabel('time')
ylabel('idx')
plt.show()

plot(t, rad2deg(y[:, 3:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
