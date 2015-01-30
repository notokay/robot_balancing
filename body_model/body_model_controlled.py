from sympy import symbols, simplify, lambdify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye, asarray
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from body_model_setup import theta1, theta2, theta3,theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque, waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants, l_leg, r_leg, crotch, body
from sympy.physics.vector import init_vprinting, vlatex
from math import fabs
init_vprinting()
import scipy
import pickle
import math
import sympy as sp


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0.12,0.01, 0.03, 0.04])
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

dt = 1./frames_per_sec

# Control
# =======

inputK = open('bm_LQR_K_useful.pkl','rb')
inputG = open('bm_gain_coefs.pkl', 'rb')
inputa1 = open('bm_angle_one_useful.pkl','rb')
inputa2 = open('bm_angle_two_useful.pkl','rb')
inputa3 = open('bm_angle_three_useful.pkl','rb')
inputa4 = open('bm_angle_four_useful.pkl','rb')
inputt = open('bm_trim_useful.pkl', 'rb')

K = pickle.load(inputK)
gain_coefs = pickle.load(inputG)
a1 = pickle.load(inputa1)
a1 = np.asarray(a1, dtype = float)
a2 = pickle.load(inputa2)
a2 = np.asarray(a2, dtype = float)
a3 = pickle.load(inputa3)
a3 = np.asarray(a3, dtype = float)
a4 = pickle.load(inputa4)
a4 = np.asarray(a4, dtype = float)
trim = pickle.load(inputt)

inputK.close()
inputG.close()
inputa1.close()
inputa2.close()
inputa3.close()
inputa4.close()
inputt.close()

c, gt1, gt1sq, gt1cb, gt2, gt2sq, gt2cb, gt3, gt3sq,gt3cb, gt4, gt4sq,gt4cb, go1, go1sq,go1cb, go2, go2sq,go2cb, go3, go3sq,go3cb, go4, go4sq,go4cb, t1, t2, t3, t4, o1, o2, o3, o4 = dynamicsymbols('c, gt1, gt1sq,gt1cb, gt2, gt2sq,gt2cb, gt3, gt3sq,gt3cb, gt4, gt4sq,gt4cb, go1, go1sq,go1cb, go2, go2sq,go2cb, go3, go3sq,go3cb, go4, go4sq,go4cb, t1, t2, t3, t4, o1, o2, o3, o4')



A = c + gt1*t1 + gt1sq*t1**2 + gt1cb*t1**3 + gt2*t2 + gt2sq*t2**2 + gt2cb*t2**3 + gt3*t3 + gt3sq*t3**2 + gt3cb*t3**3 + gt4*t4 + gt4sq*t4**2 + gt4cb*t4**3 + go1*o1 + go1sq*o1**2 + go1cb*o1**3 + go2*o2 + go2sq*o2**2 + go2cb*o2**3 + go3*o3 + go3sq*o3**2 + go3cb*o3**3+ go4*o4 + go4sq*o4**2 + go4cb*o4**3

gain_funcs = [0,0,0,0,0,0,0,0]

for element in gain_coefs:
  for coefs in element:
    d = dict(zip([c, gt1, gt1sq, gt1cb, gt2, gt2sq, gt2cb, gt3, gt3sq, gt3cb, gt4, gt4sq, gt4cb, go1, go1sq, go1cb, go2, go2sq, go2cb, go3, go3sq, go3cb, go4, go4sq, go4cb], coefs))
    func = A.subs(d)
    gain_funcs.append(lambdify((t1, t2, t3, t4, o1, o2, o3, o4), func))

torque_vector = []
lastk = []
idx_vector = []
lastidx = 0
counter = 0
time_vector = []
output_vector = []
diff_vector = []
limits_vector = []
passivity_vector = []
p_energy_vector = []
k_energy_vector = []
max_legs_pe = 0.8*20*9.8 + (0.8+0.05)*20*9.8 + 20*0.8*9.8
max_body_pe = 90*(0.8+0.15+0.5)*9.8
lpe = l_leg.potential_energy.subs(parameter_dict)
cpe = crotch.potential_energy.subs(parameter_dict)
bpe = body.potential_energy.subs(parameter_dict)
rpe = r_leg.potential_energy.subs(parameter_dict)
legs_tpe = lpe + cpe + rpe
tor1 = []
tor2 = []
tor3 = []
tor4 = []
vel_vec = []
lhip_vel = []
rhip_vel = []
ang_vec = []
int_lhip_vel = []
int_rhip_vel = []
lastreturnval = []
com_counter = 0
temptorvec = []
com_vec = []
int_com_vec = []
legs_tpe_f = lambdify((theta1, theta2, theta3, theta4), legs_tpe)
bpe_f = lambdify((theta1, theta2, theta3, theta4), bpe)
x_vec = []
lastwaisttor = 0.0
pure_torvec = []
gains_vec = []
error_vec = []
K_func_gains = []
allowed_tor = []

def getK(x):
  global gain_funcs
  K_gains = [[0.,0.,0.,0.,0.,0.,0.,0.],[1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.]]
  K_gains = asarray(K_gains)
  for i in np.arange(len(K_gains)):
    for j in np.arange(len(K_gains[i])):
      if(K_gains[i][j] != 0.0):
        K_gains[i][j] = gain_funcs[i*8+j](x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
  return K_gains
  



def calc_com(t1, t2, t3, t4):
  LH_x = -1*numerical_constants[12]*sin(t1)
  C_x = LH_x + (numerical_constants[4]/2)*cos(t1+t2)
  W_x = C_x + numerical_constants[8]*cos(t1+t2+1.57)
  C_cx = C_x + numerical_constants[5]*cos(t1+t2+1.57)
  B_x = W_x + numerical_constants[9]*cos(t1+t2+1.57+t4)
  RH_x = LH_x + numerical_constants[4]*cos(t1+t2)
  RA_x = RH_x + numerical_constants[12]*sin(t1+t2+t3)
  return ((20*LH_x + 20*C_cx + 90*B_x + RA_x*20)/150)


def continuous_energy_controller_integrator(x,t):
  global counter
  global lastwaisttor
  global lastreturnval
  global com_counter
  global gain_funcs
  global K_func_gains
  a1_error = [b*b for b in a1 - x[0]]
  a2_error = [b*b for b in a2 - x[1]]
  a3_error = [b*b for b in a3 - x[2]]
  a4_error = [b*b for b in a4 - x[3]]
  tot_error = [a+b+c+d for a, b, c, d in zip(a1_error, a2_error, a3_error, a4_error)]
  tot_error = np.array(tot_error)
  min_errors = np.where(tot_error == tot_error.min())
  closest_equil = min_errors[0][0]
  x0 = x[0] - a1[closest_equil]
  x1 = x[1] - a2[closest_equil]
  x2 = x[2] - a3[closest_equil]
  x3 = x[3] - a4[closest_equil]
  error_vec.append([x0, x1, x2, x3])
  idx_vector.append(closest_equil)

  torquelim = 120
  ankle_torquelim = 90
  limit_torque = 15*torquelim
  time_vector.append(t)
  x_vec.append(x)
  vel_vec.append([x[4], x[5], x[6], x[7]])
  ang_vec.append([x[0], x[1], x[2], x[3]])
  lhip_vel.append(x[5])
  rhip_vel.append(x[6])
  com = calc_com(x[0], x[1], x[2], x[3])
  com_vec.append(com)
  int_com_error = scipy.integrate.trapz(com_vec, time_vector)
  int_com_vec.append(int_com_error)  
  lhip_vel_error = scipy.integrate.trapz(lhip_vel, time_vector)
  int_lhip_vel.append(lhip_vel_error)
  error_x = [x0,x1,x2,x3,x[4],x[5],x[6],x[7]]
  if(counter%1 == 0):
    K_func_gains = getK(error_x)
    counter = counter + 1

#  if(x[4] < 0.3 and x[4] > -0.3):
#    if(counter%1==0):
#      K_func_gains = getK(error_x)
#      print("stay zero adapt")
#    else:
#      print("stay zero")
#  elif(x[4] < 2. and x[4] > -2.):
#    if(counter %1==0):
#      K_func_gains = getK(error_x)
#      print("adapt")
#    else:
#      print('stay')
#  else:
#    print("hispeed")
  returnval = 1*dot(K_func_gains, [x0,x1,x2,x3, x[4],x[5],x[6],x[7]])
  returnval[0] = 0. - returnval[0]
  returnval[1] = trim[closest_equil][0] - returnval[1]
  returnval[2] = trim[closest_equil][1] - returnval[2]
  returnval[3] = trim[closest_equil][2] - returnval[3]
  pure_tor = 1* dot(K_func_gains, [x0,x1,x2,x3,x[4],x[5],x[6],x[7]])
  pure_tor[0] = 0. - pure_tor[0]
  pure_tor[1] = trim[closest_equil][0] - pure_tor[1]
  pure_tor[2] = trim[closest_equil][1] - pure_tor[2]
  pure_tor[3] = trim[closest_equil][2] - pure_tor[3]
  pure_torvec.append(pure_tor)
  coord_dict = dict(zip(coordinates, (x[0], x[1], x[2], x[3])))
  com = calc_com(x[0], x[1], x[2], x[3])
  returnval[0] = 900*com

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
  
  if(x[1] >  0.516):
    returnval[1] = returnval[1] + -1*limit_torque*(x[1] - 0.516)
  if(x[1] < -0.6):
    returnval[1] = returnval[1] + -1*limit_torque*(x[1] - 0.6)
  if(x[2] > 0.516):
    returnval[2] = returnval[2] + -1*limit_torque*(x[2] - 0.516)
  if(x[2] < -0.6):
    returnval[2] = returnval[2] + -1*limit_torque*(x[2] - 0.6)
  if(x[3] > 0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
  if(x[3] < -1*0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)

  if(returnval[0] > ankle_torquelim):
    returnval[0] = ankle_torquelim
  if(returnval[0] < -1*ankle_torquelim):
    returnval[0] = -1*ankle_torquelim
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
  
  returnval[1] = returnval[1] + 10*com + 10*int_com_error  + -50*(1/(fabs(com)+0.1))*x[5]
  returnval[0] = returnval[0] + 10*int_com_error + 1*com*int_com_error + -10*(1/(fabs(com)+0.1))*x[0]
  returnval[3] = returnval[3] + -1*(1/(fabs(com)+0.1))*x[7] + -250*x[7]
  returnval[2] = returnval[2] +  -10*fabs(x[4]*x[7])*x[6]  + -100*x[6] + -1*(1/(fabs(com)+0.1))*x[6]
  if(x[1] >  0.516):
    returnval[1] = returnval[1] + -1*limit_torque*(x[1] - 0.516)
  if(x[1] < -0.6):
    returnval[1] =  returnval[1] + -1*limit_torque*(x[1] - 0.6)
  if(x[2] > 0.516):
    returnval[2] = returnval[2] + -1*limit_torque*(x[2] - 0.516)
  if(x[2] < -0.6):
    returnval[2] = returnval[2] + -1*limit_torque*(x[2] - 0.6)
  if(x[3] > 0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
  if(x[3] < -1*0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
  if(returnval[0] > ankle_torquelim):
    returnval[0] = ankle_torquelim
  if(returnval[0] < -1*ankle_torquelim):
    returnval[0] = -1*ankle_torquelim
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

  counter = counter + 1
  torque_vector.append(returnval)
  return returnval

args['specified'] = continuous_energy_controller_integrator

y = odeint(right_hand_side, x0, t, args=(args,))

#Set up simulation


LH_x = -1*numerical_constants[0]*sin(y[:,0])
la_x = -1*numerical_constants[12]*sin(y[:,0])
LH_y = numerical_constants[0]*cos(y[:,0])

C_x = LH_x + (numerical_constants[4]/2)*cos(y[:,1] + y[:,0])
C_y = LH_y + (numerical_constants[4]/2)*sin(y[:,1] + y[:,0])

W_x = C_x + numerical_constants[8]*cos(y[:,1] + y[:,0] + 1.57)
W_y = C_y + numerical_constants[8]*sin(y[:,1] + y[:,0] + 1.57)
c_x = C_x + numerical_constants[5]*cos(y[:,1] + y[:,0] + 1.57)

B_x = W_x + numerical_constants[9]*2*cos(y[:,1] + y[:,0] + 1.57 + y[:,3])
B_y = W_y + numerical_constants[9]*2*sin(y[:,1] + y[:,0] + 1.57 + y[:,3])
b_x = W_x + numerical_constants[9]*cos(y[:,1] + y[:,0] + 1.57 + y[:,3])

RH_x = LH_x + numerical_constants[4]*cos(y[:,1] + y[:,0])
RH_y = LH_y + numerical_constants[4]*sin(y[:,1] + y[:,0])

RA_x = RH_x + numerical_constants[12]*2*sin(y[:,2] + y[:,1] + y[:,0])
ra_x = RH_x + numerical_constants[12]*sin(y[:,2] + y[:,1] + y[:,0])
RA_y = RH_y + -1*numerical_constants[12]*2*cos(y[:,2] + y[:,1] + y[:,0])

com_loc = []
for l, c, b, r in zip(la_x, c_x, b_x, ra_x):
  com_loc.append((20*l+20*c+50*b+20*r)/110)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,aspect='equal', xlim = (-2, 2), ylim = (-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
circles, = ax.plot([], [], 'bo', ms = 10)
time_template = 'time=%.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
  line.set_data([],[])
  circles.set_data([],[])
  time_text.set_text('')
  return circles, line, time_text

def animate(i):
  thisx = [0, LH_x[i], C_x[i], W_x[i], B_x[i], W_x[i], C_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LH_y[i], C_y[i], W_y[i], B_y[i], W_y[i], C_y[i], RH_y[i], RA_y[i]]
  circles.set_data(com_loc[i], 0)

  line.set_data(thisx, thisy)
  time_text.set_text(time_template% (i*dt))
  return circles, line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=dt*1000, blit=True, init_func=init)
#ani.save('body_model_com_integrator.mp4')
plt.show()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

pe, pe_tot, ke = dynamicsymbols('pe, pe_tot, ke')
energies = [ke,pe]

ax1.plot(time_vector, ang_vec)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle[deg]')
ax1.legend(["${}$".format(vlatex(c)) for c in coordinates])

ax3.plot(time_vector, torque_vector)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Angle torques')
ax3.legend(["${}$".format(vlatex(c)) for c in specified])

ax4.plot(time_vector, pure_torvec)
ax4.set_xlabel('Time[s]')
ax4.set_ylabel('Pure Angle torques')
ax4.legend(["${}$".format(vlatex(c)) for c in specified])

ax5.plot(time_vector, idx_vector)
ax5.set_xlabel('time')
ax5.set_ylabel('idx')

ax2.plot(time_vector,vel_vec)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular Rate [deg/s]')
ax2.legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
