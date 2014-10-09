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
from double_block_setup import theta1, theta2,  omega1, omega2,l_ankle_torque, l_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()
import pickle


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0,0])
#1.495,3.13
#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0])}

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

Q = ((1/0.6)**2)*eye(4)
R = eye(2)

S = solve_continuous_are(A, B, Q, R)
gainK = dot(dot(inv(R), B.T), S)

inputK = open('double_block_LQR_K.pkl', 'rb')
inputa1 = open('double_block_angle_1_zoom.pkl','rb')
inputa2 = open('double_block_angle_2_zoom.pkl','rb')
inputt = open('double_block_trim.pkl','rb')

K = pickle.load(inputK)
angle_1 = pickle.load(inputa1)
angle_1 = np.asarray(angle_1, dtype = float)
angle_2 = pickle.load(inputa2)
angle_2 = np.asarray(angle_2, dtype = float)
trim = pickle.load(inputt)
trim = np.asarray(trim, dtype = float)

inputK.close()
inputa1.close()
inputa2.close()
inputt.close()

torque_vector = []
idx_vector = []
lastidx = 0
counter = 0
tracking_vector = []
time_vector = []
output_vector = []
diff_vector = []

def stay_controller(x,t):
  global lastidx
  global counter
  torquelim = 900
  if(counter == 0):
    lastidx = np.abs(angle_1 - x[0]).argmin()
    counter = counter + 1
    print('first round')
    print(lastidx)
  if(x[2] < 100 and x[2] > -100):
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
  if(x[0] > 1 or x[0] < -1):
    returnval[1] = 0
  returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
def trim_controller(x,t):
  idx = np.abs(angle_2 - x[1]).argmin()
  return [0,trim[idx]]

#args['specified'] = stay_controller

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LA_x = numerical_constants[0]*sin(y[:,0])
LA_y = numerical_constants[0]*cos(y[:,0])

B_x = LA_x + numerical_constants[11]*sin(y[:,1] + y[:,0])
B_y = LA_y + numerical_constants[11]*cos(y[:,1] + y[:,0])

RH_x = LA_x + numerical_constants[4]*sin(y[:,0] + y[:,1] + 1.57)
RH_y = LA_y + numerical_constants[4]*cos(y[:,0] + y[:,1] + 1.57)

RA_x = RH_x + numerical_constants[8]*2*sin(3.14 + y[:,1] + y[:,0])
RA_y = RH_y + numerical_constants[8]*2*cos(3.14 + y[:,1] + y[:,0])

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
  thisx = [0, LA_x[i],B_x[i], LA_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LA_y[i],B_y[i], LA_y[i], RH_y[i], RA_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%(i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('double_block_controlled.mp4')
plt.show()

plot(t, rad2deg(y[:,:2]))
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

plot(t, rad2deg(y[:, 2:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
