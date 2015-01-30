from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import fabs
import matplotlib.animation as animation
from matplotlib import cm
from body_model_setup import theta1, theta2, theta3,theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque,waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants
import pickle

#from utils import controllable

init_vprinting()
rcParams['figure.figsize'] = (14.0, 6.0)

inputa1 = open('bm_angle_one.pkl','rb')
inputa2 = open('bm_angle_two.pkl','rb')
inputa3 = open('bm_angle_three.pkl','rb')
inputa4 = open('bm_angle_four.pkl', 'rb')
inputt = open('bm_trim.pkl', 'rb')

a1 = pickle.load(inputa1)
a2 = pickle.load(inputa2)
a3 = pickle.load(inputa3)
a4 = pickle.load(inputa4)
trim = pickle.load(inputt)

inputa1.close()
inputa2.close()
inputa3.close()
inputa4.close()
inputt.close()


LH_x = []
LH_y = []
C_x = []
C_y = []
W_x = []
W_y = []
B_x = []
B_y = []
RH_x = []
RH_y = []
RA_x = []
RA_y = []
angle_1 = []
angle_2 = []
angle_3 = []
angle_4 = []
trim_vec = []
left_leg = numerical_constants[0]
hip_width = numerical_constants[4]
right_leg = numerical_constants[0]
body_height = numerical_constants[8]
body_com_height = numerical_constants[9]
"""
for i in range(len(a1)):
  if(i%5==0):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
    angle_4.append(a4[i])

a1 = angle_1
a2 = angle_2
a3 = angle_3
a4 = angle_4

angle_1 =[]
angle_2 = []
angle_3 = []
angle_4 = []
"""
for i in range(len(a1)):
  if(a1[i] > 0.0 and a1[i] < 0.2):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
    angle_4.append(a4[i])
    trim_vec.append(trim[i])

a1 = angle_1
a2 = angle_2
a3 = angle_3
a4 = angle_4
trim = trim_vec

angle_1 =[]
angle_2 = []
angle_3 = []
angle_4 = []
trim_vec = []

for i in range(len(a1)):
  if(a2[i] > -0.0  and a2[i] < 0.51):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
    angle_4.append(a4[i])
    trim_vec.append(trim[i])

a1 = angle_1
a2 = angle_2
a3 = angle_3
a4 = angle_4
trim = trim_vec

angle_1 =[]
angle_2 = []
angle_3 = []
angle_4 = []
trim_vec = []

for i in range(len(a1)):
  if(a3[i] > -0.0 and a3[i] < 0.51):
    angle_1.append(a1[i])
    angle_2.append(a2[i])
    angle_3.append(a3[i])
    angle_4.append(a4[i])
    trim_vec.append(trim[i])
a1 = angle_1
a2 = angle_2
a3 = angle_3
a4 = angle_4
trim = trim_vec

angle_1 = []
angle_2 = []
angle_3 = []
angle_4 = []
trim_vec = []
for i in range(len(a1)):
  if(a4[i] > - 0.4 and a4[i] < 0.4):
      angle_1.append(a1[i])
      angle_2.append(a2[i])
      angle_3.append(a3[i])
      angle_4.append(a4[i])
      trim_vec.append(trim[i])
a1 = angle_1
a2 = angle_2
a3 = angle_3
a4 = angle_4
trim = trim_vec

for i in range(len(a1)):
  LH_x.append(-1*left_leg*sin(a1[i]))
  LH_y.append(left_leg*cos(a1[i]))
  C_x.append(-1*left_leg*sin(a1[i]) + (hip_width/2)*cos(a1[i] + a2[i]))
  C_y.append(left_leg*cos(a1[i]) + (hip_width/2)*sin(a1[i] + a2[i]))
  W_x.append(-1*left_leg*sin(a1[i]) + (hip_width/2)*cos(a1[i] + a2[i]) + body_height*cos(a1[i] + a2[i] + 1.57))
  W_y.append(left_leg*cos(a1[i]) + (hip_width/2)*sin(a1[i] + a2[i]) + body_height*sin(a1[i] + a2[i] + 1.57))
  B_x.append(-1*left_leg*sin(a1[i]) + (hip_width/2)*cos(a1[i] + a2[i]) + body_height*cos(a1[i] + a2[i] + 1.57) + body_com_height*2*cos(a1[i] + a2[i] + 1.57 + a4[i]))
  B_y.append(left_leg*cos(a1[i]) + (hip_width/2)*sin(a1[i] + a2[i]) + body_height*sin(a1[i] + a2[i] + 1.57) + body_com_height*2*sin(a1[i] + a2[i] + 1.57 + a4[i]))
  RH_x.append(-1*left_leg*sin(a1[i]) + hip_width*cos(a2[i] + a1[i]))
  RH_y.append(left_leg*cos(a1[i]) + hip_width*sin(a2[i] + a1[i]))
  RA_x.append(-1*left_leg*sin(a1[i]) + hip_width*cos(a2[i] + a1[i]) + right_leg*sin(a1[i]+a2[i]+a3[i]))
  RA_y.append(left_leg*cos(a1[i]) + hip_width*sin(a2[i] + a1[i]) + -1*right_leg*cos(a1[i] + a2[i] + a3[i]))

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, aspect = 'equal',  xlim = (-2,2), ylim = (-0.5,2))

for i in range(len(LH_x)):
  thisx = [0, LH_x[i],C_x[i], W_x[i], B_x[i],W_x[i], C_x[i], RH_x[i], RA_x[i]]
  thisy = [0, LH_y[i], C_y[i], W_y[i], B_y[i], W_y[i], C_y[i], RH_y[i], RA_y[i]]
  plt.plot(thisx,thisy)

plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = a1
ax.scatter(a1, a2, a3, c=c)
ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')
plt.show()

outputa1 = open('bm_angle_one_useful.pkl', 'wb')
outputa2 = open('bm_angle_two_useful.pkl','wb')
outputa3 = open('bm_angle_three_useful.pkl','wb')
outputa4 = open('bm_angle_four_useful.pkl', 'wb')
outputt = open('bm_trim_useful.pkl','wb')

pickle.dump(a1, outputa1)
pickle.dump(a2, outputa2)
pickle.dump(a3, outputa3)
pickle.dump(a4, outputa4)
pickle.dump(trim, outputt)


outputa1.close()
outputa2.close()
outputa3.close()
outputa4.close()
outputt.close()
