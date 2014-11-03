from sympy import symbols, simplify, lambdify
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
from triple_pendulum_setup_alt import theta1, theta2, theta3, omega1, omega2, omega3, l_ankle_torque, l_hip_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants, ke_lleg, ke_rleg, ke_body, r_leg, body, l_leg
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

initial_coordinates = array([0.1,0.22,0.2])
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
p_energy_vector = []
k_energy_vector = []
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
max_pe = 0.8*30*9.8+(0.8+0.5)*90.0*9.8+30.0*(0.8+cos(0.6)*0.4)*9.8
last1 = 0.0
last2 = 0.0
last1vec = []

lpe = l_leg.potential_energy.subs(parameter_dict)
bpe = body.potential_energy.subs(parameter_dict)
rpe = r_leg.potential_energy.subs(parameter_dict)

tpe = lpe+bpe+rpe

tpe_f = lambdify((theta1,theta2,theta3), tpe)
allowed_tor = []

def calc_com(t1, t2, t3):
  LA_x = -1*numerical_constants[0]*sin(t1)
  C_x = LA_x + numerical_constants[5]*cos(t1+t2)
  RH_x = LA_x + numerical_constants[4]*cos(t1+t2)
  M_x = C_x + numerical_constants[11]*cos(t1+t2+1.57)
  RA_x = RH_x + numerical_constants[8]*2*sin(t1+t2+t3)
  return (30*LA_x+90*M_x+30*RA_x)/150

def controller(x, t):
  torque_vector.append([200*sin(t), -100*sin(t),50*sin(t)] )
  time_vector.append(t)
  return [200*sin(t), -100*sin(t), 50*sin(t)]

def limits_controller(x,t):
  global lastidx
  global counter
  global last1
  global last2
  torquelim = 180
  limit_torque = 300
  com = calc_com(x[0], x[1], x[2])

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
  coord_dict = dict(zip(coordinates, (x[0],x[1],x[2])))
  tp = tpe_f(x[0], x[1],x[2])
  p_energy_vector.append(tp)
  ke = max_pe - tp
  k_energy_vector.append(ke)
  allowtor = (ke/(2*fabs(x[4])), ke/(2*fabs(x[5])))
  allowed_tor.append(allowtor)
  
  if(returnval[1] > allowtor[0]):
    returnval[1] = allowtor[0]
  if(returnval[1] < -1*allowtor[0]):
    returnval[1] = -1*allowtor[0]
  if(returnval[2] > allowtor[1]):
    returnval[2] = allowtor[1]
  if(returnval[2] < -1*allowtor[1]):
    returnval[2] = -1*allowtor[1]
  if(returnval[1] > torquelim): 
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
#  returnval[0] = 500*com
  if(x[0] < -0.5):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.5):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
    
  if(x[1] >  0.516):
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
  if(x[1] < -0.516):
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
  if(x[2] > 0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
  if(x[2] < -0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
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

com_loc = []
for l, b, r in zip(LA_x, M_x, RA_x):
  com_loc.append((30*l+90*b+30*r)/150)

dt = 0.05

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,aspect='equal', xlim = (-2, 2), ylim = (-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
circles, = ax.plot([],[], 'bo', ms=10)

time_template = 'time=%.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
  line.set_data([],[])
  circles.set_data([],[])
  time_text.set_text('')
  return circles,line, time_text

def animate(i):
  thisx = [0, LA_x[i], RH_x[i], RA_x[i], RH_x[i], C_x[i], M_x[i]]
  thisy = [0, LA_y[i], RH_y[i], RA_y[i], RH_y[i], C_y[i], M_y[i]]
  circles.set_data(com_loc[i],0)

  line.set_data(thisx, thisy)
  time_text.set_text(time_template%((i*dt)/3))
  return circles, line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('triple_pendulum_energy.mp4')
plt.show()

f, (ax1, ax2,ax3, ax4, ax5) = plt.subplots(5)

pe, pe_tot, ke = dynamicsymbols('pe, pe_tot, ke')
energies = [ke, pe]

ax3.plot(time_vector, k_energy_vector)
ax3.plot(time_vector, p_energy_vector)
ax3.set_xlabel('Time[s]')
ax3.set_ylabel('Energy')
ax3.legend(["${}$".format(vlatex(e)) for e in energies])

ax1.plot(t, rad2deg(y[:,:3]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle[deg]')
ax1.legend(["${}$".format(vlatex(c)) for c in coordinates])

ax4.plot(time_vector, torque_vector)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Angle torques')
ax4.legend(["${}$".format(vlatex(c)) for c in specified])

ax5.plot(time_vector, idx_vector)
ax5.set_xlabel('time')
ax5.set_ylabel('idx')

ax2.plot(t, rad2deg(y[:, 3:]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular Rate [deg/s]')
ax2.legend(["${}$".format(vlatex(s)) for s in speeds])

plt.show()
