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
from triple_pendulum_setup import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = deg2rad(90.0) * array([2, 2, 2])

#initial_speeds = deg2rad(-5.0) * ones(len(speeds))
initial_speeds = zeros(len(speeds))

x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 10.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

#Create dictionaries for the values for equilibrium point
equilibrium_point = zeros(len(coordinates + speeds))
equilibrium_point[:3] = deg2rad(180.0)
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))

#Jacobian of the forcing vector w.r.t. states and inputs
F_A = forcing_vector.jacobian(coordinates + speeds)
F_B = forcing_vector.jacobian(specified)

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

Q = 1000*eye(6)
R = eye(3)

S = solve_continuous_are(A, B, Q, R)
K = dot(dot(inv(R), B.T), S)

torque_vector = []
time_vector = []
output_vector = []
diff_vector = []

def controller(x, t):
  torque_vector.append([200*sin(t), -100*sin(t),50*sin(t)] )
  time_vector.append(t)
  return [200*sin(t), -100*sin(t), 50*sin(t)]
def good_controller(x, t):
  temp = -dot(K,x)
#  temp[0] = 0
  torque_vector.append(temp)
  time_vector.append(t)
  return temp

def pi_controller(x, t):
  desired = deg2rad(90.0)*array([2,2,2])
  diff = desired - x[:3]
 #diff[0] = diff[0]*600
  #diff[1] = diff[1]*200
  #diff[2] = diff[2]*100
  torque_vector.append(diff)
  time_vector.append(t)
  return diff*200

#args['specified'] = good_controller

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LA_x = numerical_constants[0]*sin(y[:,0])
LA_y = numerical_constants[0]*cos(y[:,0])

RH_x = LA_x + numerical_constants[4]*sin(y[:,1])
RH_y = LA_y + numerical_constants[4]*cos(y[:,1])

RA_x = RH_x + numerical_constants[8]*2*sin(y[:,2])
RA_y = RH_y + numerical_constants[8]*2*cos(y[:,2])

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
  thisx = [0, LA_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LA_y[i], RH_y[i], RA_y[i]]

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%(i*dt))
  return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('triple_pendulum.mp4')
plt.show()

plot(t, rad2deg(y[:,:3]))
xlabel('Time [s]')
ylabel('Angle[deg]')
legend(["${}$".format(vlatex(c)) for c in coordinates])
plt.show()

plot(time_vector, torque_vector)
xlabel('Time [s]')
ylabel('Angle torques')
plt.show()

plot(t, rad2deg(y[:, 3:]))
xlabel('Time [s]')
ylabel('Angular Rate [deg/s]')
legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
