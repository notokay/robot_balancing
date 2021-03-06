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


rcParams['figure.figsize'] = (14.0, 6.0)

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
# Specify Numerical Quantities
# ============================

initial_coordinates = array([0.1,0.1,0.0,0.0])
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
time_vector = []
output_vector = []
diff_vector = []
limits_vector = []
passivity_vector = []
p_energy_vector = []
k_energy_vector = []
max_legs_pe = 0.8*20*9.8 + (0.8+0.05)*20*9.8 + 20*1.6*9.8
max_body_pe = 50*(0.8+0.15+0.5)*9.8
lpe = l_leg.potential_energy.subs(parameter_dict)
cpe = crotch.potential_energy.subs(parameter_dict)
bpe = body.potential_energy.subs(parameter_dict)
rpe = r_leg.potential_energy.subs(parameter_dict)
legs_tpe = lpe + cpe + rpe
tor1 = []
tor2 = []
tor3 = []
tor4 = []
lastvel = []
temptorvec = []
com_vec = []
int_com_vec = []
legs_tpe_f = lambdify((theta1, theta2, theta3, theta4), legs_tpe)
bpe_f = lambdify((theta1, theta2, theta3, theta4), bpe)

allowed_tor = []
"""
a1, a2, a3, a4, c = dynamicsymbols('a1, a2, a3,a4, c')

LH_x = -1*numerical_constants[12]*sp.cos(a1)
C_x = LH_x + (numerical_constants[4]/2)*sp.cos(a1+a2)
W_x = C_x + numerical_constants[8]*sp.cos(a1+a2+1.57)
C_cx = C_x + numerical_constants[5]*sp.cos(a1+a2+1.57)
B_x = W_x + numerical_constants[9]*sp.cos(a1+a2+1.57+a4)
RH_x = LH_x + numerical_constants[4]*sp.cos(a1+a2)
RA_x = RH_x + numerical_constants[12]*sp.sin(a1+a2+a3)
com = (20*LH_x + 20*C_cx + 50*B_x + RA_x*20)/150
com_solved = solve(com-c, c)
com_solved_f = lambdify((a1, a2, a3), com_solved)
"""
def calc_com(t1, t2, t3, t4):
  LH_x = -1*numerical_constants[12]*sin(t1)
  C_x = LH_x + (numerical_constants[4]/2)*cos(t1+t2)
  W_x = C_x + numerical_constants[8]*cos(t1+t2+1.57)
  C_cx = C_x + numerical_constants[5]*cos(t1+t2+1.57)
  B_x = W_x + numerical_constants[9]*cos(t1+t2+1.57+t4)
  RH_x = LH_x + numerical_constants[4]*cos(t1+t2)
  RA_x = RH_x + numerical_constants[12]*sin(t1+t2+t3)
  return ((20*LH_x + 20*C_cx + 50*B_x + RA_x*20)/110)

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
  torquelim = 300
  limit_torque = 2*torquelim
  com = calc_com(x[0], x[1], x[2], x[3])
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
  global lastidx
  global counter
  torquelim = 300
  limit_torque = 2*torquelim
  com = calc_com(x[0], x[1], x[2], x[3])
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
      print("stay")
      print(idx)
      idx_vector.append(lastidx)
    else:
      idx = np.abs(a1-x[0]).argmin()
      idx = idx + np.abs(a2[idx:idx+40] - x[1]).argmin()
      idx = idx + np.abs(a4[idx:idx+40] - x[3]).argmin()
      lastidx = idx
      print(idx)
      idx_vector.append(lastidx)
  if(idx > (len(K)-1)):
    idx = len(K)-1
  if(idx < 0):
    idx = 0
  returnval = -dot(K[idx], x)
  coord_dict = dict(zip(coordinates, (x[0], x[1], x[2], x[3])))
  tlp = legs_tpe_f(x[0], x[1], x[2], x[3])
  tbp = bpe_f(x[0], x[1], x[2], x[3])
  p_energy_vector.append(tlp)
  ke_legs = max_legs_pe - tlp
  ke_body = max_body_pe - tbp
  k_energy_vector.append(ke_legs)
  allowtor = ( ke_legs/(2*fabs(x[5])+0.000001), ke_legs/(2*fabs(x[6])+0.000001), ke_body/(fabs(x[7])+0.000001))
  allowed_tor.append(allowtor)
  com = calc_com(x[0], x[1], x[2], x[3])
  returnval[0] = 2000*com
  if(returnval[1] > allowtor[0]):
    returnval[1] = allowtor[0]
  if(returnval[1] < -1*allowtor[0]):
    returnval[1] = -1*allowtor[0]
  if(returnval[2] > allowtor[1]):
    returnval[2] = allowtor[1]
  if(returnval[2] < -1*allowtor[1]):
    returnval[2] = -1*allowtor[1]
  if(returnval[3] > allowtor[2]):
    returnval[3] = allowtor[2]
  if(returnval[3] < -1*allowtor[2]):
    returnval[3] = -1*allowtor[2]

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
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1]-0.516)
  if(x[1] < -0.516):
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1] - 0.516)
  if(x[2] > 0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[2] < -0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[3] > 0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3]-0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
  if(x[3] < -1*0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
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

  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval
def energy_controller_2(x,t):
  global lastidx
  global counter
  torquelim = 180
  ankle_torquelim = 90
  limit_torque = 8*torquelim
  com = calc_com(x[0], x[1], x[2], x[3])
  if(counter==0):
    lastidx = np.abs(a1 - x[0]).argmin()
    lastidx = lastidx + np.abs(a2[lastidx:lastidx+40] - x[1]).argmin()
    lastidx = lastidx + np.abs(a4[lastidx:lastidx+40] - x[3]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print (lastidx)
  if(x[4] < 1.0 and x[4] > -1.0):
    idx = np.abs(a1 - x[0]).argmin()
    idx = idx + np.abs(a2[idx:idx+40] - x[1]).argmin()
    idx = idx + np.abs(a4[idx:idx+40] - x[3]).argmin()
    lastidx = idx
    idx_vector.append(idx)
    print('adapt')
    print(idx)
  else:
    idx = lastidx
    print("stay")
    print(idx)
    idx_vector.append(lastidx)

  if(idx > (len(K)-1)):
    idx = len(K)-1
  if(idx < 0):
    idx = 0
  returnval = -dot(K[idx], x)
  coord_dict = dict(zip(coordinates, (x[0], x[1], x[2], x[3])))
  tlp = legs_tpe_f(x[0], x[1], x[2], x[3])
  tbp = bpe_f(x[0], x[1], x[2], x[3])
  p_energy_vector.append(tlp)
  ke_legs = max_legs_pe - tlp
  ke_body = max_body_pe - tbp
  k_energy_vector.append(ke_legs)
  allowtor = ( ke_legs/(2*fabs(x[5])+0.000001), ke_legs/(2*fabs(x[6])+0.000001), ke_body/(fabs(x[7])+0.000001))
  allowed_tor.append(allowtor)
  com = calc_com(x[0], x[1], x[2], x[3])
  returnval[0] = 800*com

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
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1]-0.516)
  if(x[1] < -0.516):
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1] - 0.516)
  if(x[2] > 0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[2] < -0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[3] > 0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3]-0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
  if(x[3] < -1*0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
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
  if(returnval[1] > allowtor[0]):
    returnval[1] = allowtor[0]
  if(returnval[1] < -1*allowtor[0]):
    returnval[1] = -1*allowtor[0]
  if(returnval[2] > allowtor[1]):
    returnval[2] = allowtor[1]
  if(returnval[2] < -1*allowtor[1]):
    returnval[2] = -1*allowtor[1]
  if(returnval[3] > allowtor[2]):
    returnval[3] = allowtor[2]
  if(returnval[3] < -1*allowtor[2]):
    returnval[3] = -1*allowtor[2]
  com_vec.append(com)
  time_vector.append(t)
  int_com_error = scipy.integrate.trapz(com_vec, time_vector)
  int_com_vec.append(int_com_error)
  returnval[3] = returnval[3] + 100*int_com_error
  returnval[0] = returnval[0] + 100*int_com_error
  if(returnval[0] > ankle_torquelim):
    returnval[0] = ankle_torquelim
  if(returnval[0] < -1*ankle_torquelim):
    returnval[0] = -1*ankle_torquelim
  if(returnval[3] < -1*torquelim):
    returnval[3] = -1*torquelim
  if(returnval[3] > torquelim):
    returnval[3] = torquelim
  if(returnval[3] > allowtor[2]):
    returnval[3] = allowtor[2]
  if(returnval[3] < -1*allowtor[2]):
    returnval[3] = -1*allowtor[2]

  torque_vector.append(returnval)
  return returnval

def energy_controller_integrator(x,t):
  global lastidx
  global counter
  torquelim = 180
  ankle_torquelim = 90
  limit_torque = 8*torquelim
  com = calc_com(x[0], x[1], x[2], x[3])
  if(counter==0):
    lastidx = np.abs(a1 - x[0]).argmin()
    lastidx = lastidx + np.abs(a2[lastidx:lastidx+40] - x[1]).argmin()
    lastidx = lastidx + np.abs(a4[lastidx:lastidx+40] - x[3]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print (lastidx)
  if(x[4] < 1.0 and x[4] > -1.0):
    idx = np.abs(a1 - x[0]).argmin()
    idx = idx + np.abs(a2[idx:idx+40] - x[1]).argmin()
    idx = idx + np.abs(a4[idx:idx+40] - x[3]).argmin()
    lastidx = idx
    idx_vector.append(idx)
    print('adapt')
    print(idx)
  else:
    idx = lastidx
    print("stay")
    print(idx)
    idx_vector.append(lastidx)

  if(idx > (len(K)-1)):
    idx = len(K)-1
  if(idx < 0):
    idx = 0
  returnval = -dot(K[idx], x)
  coord_dict = dict(zip(coordinates, (x[0], x[1], x[2], x[3])))
  tlp = legs_tpe_f(x[0], x[1], x[2], x[3])
  tbp = bpe_f(x[0], x[1], x[2], x[3])
  p_energy_vector.append(tlp)
  ke_legs = max_legs_pe - tlp
  ke_body = max_body_pe - tbp
  k_energy_vector.append(ke_legs)
  allowtor = ( ke_legs/(2*fabs(x[5])+0.000001), ke_legs/(2*fabs(x[6])+0.000001), ke_body/(fabs(x[7])+0.000001))
  allowed_tor.append(allowtor)
  com = calc_com(x[0], x[1], x[2], x[3])
  returnval[0] = 800*com

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
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1]-0.516)
  if(x[1] < -0.516):
    returnval[1] = returnval[1]+-1*limit_torque*(x[1]-0.516)
#    returnval[1] = -1*limit_torque*(x[1] - 0.516)
  if(x[2] > 0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[2] < -0.516):
    returnval[2] = returnval[2]+-1*limit_torque*(x[2]-0.516)
#    returnval[2] = -1*limit_torque*(x[2] - 0.516)
  if(x[3] > 0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3]-0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
  if(x[3] < -1*0.6):
    returnval[3] = returnval[3] + -1*limit_torque*(x[3] - 0.6)
#    returnval[3] = -1*limit_torque*(x[3] - 0.6)
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
    
  if(returnval[1] > allowtor[0]):
    returnval[1] = allowtor[0]
  if(returnval[1] < -1*allowtor[0]):
    returnval[1] = -1*allowtor[0]
  if(returnval[2] > allowtor[1]):
    returnval[2] = allowtor[1]
  if(returnval[2] < -1*allowtor[1]):
    returnval[2] = -1*allowtor[1]
  if(returnval[3] > allowtor[2]):
    returnval[3] = allowtor[2]
  if(returnval[3] < -1*allowtor[2]):
    returnval[3] = -1*allowtor[2]
    
  com_vec.append(com)
  time_vector.append(t)
  int_com_error = scipy.integrate.trapz(com_vec, time_vector)
  int_com_vec.append(int_com_error)
  returnval[3] = returnval[3] + 100*int_com_error
  returnval[0] = returnval[0] + 100*int_com_error
  if(returnval[0] > ankle_torquelim):
    returnval[0] = ankle_torquelim
  if(returnval[0] < -1*ankle_torquelim):
    returnval[0] = -1*ankle_torquelim
  if(returnval[3] < -1*torquelim):
    returnval[3] = -1*torquelim
  if(returnval[3] > torquelim):
    returnval[3] = torquelim
  if(returnval[3] > allowtor[2]):
    returnval[3] = allowtor[2]
  if(returnval[3] < -1*allowtor[2]):
    returnval[3] = -1*allowtor[2]
  torque_vector.append(returnval)
  return returnval

args['specified'] = energy_controller_integrator

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

dt = 0.05

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
  time_text.set_text(time_template%((i*dt)/3))
  return circles, line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)
#ani.save('body_model_com_integrator.mp4')
plt.show()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

pe, pe_tot, ke = dynamicsymbols('pe, pe_tot, ke')
energies = [ke,pe]

ax1.plot(t, rad2deg(y[:,:4]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angle[deg]')
ax1.legend(["${}$".format(vlatex(c)) for c in coordinates])

ax3.plot(time_vector, torque_vector)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Angle torques')
ax3.legend(["${}$".format(vlatex(c)) for c in specified])

ax4.plot(time_vector, k_energy_vector)
ax4.plot(time_vector, p_energy_vector)
ax4.set_xlabel('Time[s]')
ax4.set_ylabel('Energy')
ax4.legend(["${}$".format(vlatex(e)) for e in energies])

ax5.plot(time_vector, idx_vector)
ax5.set_xlabel('time')
ax5.set_ylabel('idx')

ax2.plot(t, rad2deg(y[:, 4:]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular Rate [deg/s]')
ax2.legend(["${}$".format(vlatex(s)) for s in speeds])
plt.show()
