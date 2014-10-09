from sympy import symbols, simplify, trigsimp, solve, latex, diff, cos, sin
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
#from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, pi, zeros, dot, eye
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.physics.vector import init_vprinting, vlatex
from matplotlib import cm
from triple_pendulum_setup import parameter_dict, l_leg_length, hip_width, r_leg_com_length
init_vprinting()
import pickle

inputx_two = open('triple_pendulum_angle_one_lots.pkl', 'rb')
inputy_two = open('triple_pendulum_angle_two_lots.pkl', 'rb')
inputz_two = open('triple_pendulum_angle_three_lots.pkl', 'rb')
inputx_three = open('triple_pendulum_angle_one_lots.pkl', 'rb')
inputy_three = open('triple_pendulum_angle_two_lots.pkl', 'rb')
inputz_three = open('triple_pendulum_angle_three_lots.pkl', 'rb')
inputdet_two = open('triple_pen_controllability_angle_two_lots.pkl','rb')
inputdet_three = open('triple_pen_controllability_angle_three_lots.pkl','rb')

X2 = pickle.load(inputx_two)
Y2 = pickle.load(inputy_two)
Z2 = pickle.load(inputz_two)
X3 = pickle.load(inputx_three)
Y3 = pickle.load(inputy_three)
Z3 = pickle.load(inputz_three)
det_2 = pickle.load(inputdet_two)
det_3 = pickle.load(inputdet_three)

inputx_two.close()
inputy_two.close()
inputz_two.close()
inputx_three.close()
inputy_three.close()
inputz_three.close()
inputdet_two.close()
inputdet_three.close()

l_leg = parameter_dict[l_leg_length]
hip = parameter_dict[hip_width]
r_leg = parameter_dict[l_leg_length]

x2 = []
y2 = []
z2 = []
det_two = []
LA_x2 = []
LA_y2 = []
RH_x2 = []
RH_y2 = []
RA_x2 = []
RA_y2 = []

for i in range(len(det_2)):
  if(det_2[i] < 0.1):
    if(X2[i] < 1.57 and X2[i] > -1.57 ):
      x2.append(X2[i])
      y2.append(Y2[i])
      z2.append(Z2[i])
      det_two.append(det_2[i])
      LA_x2.append(l_leg*sin(X2[i]))
      LA_y2.append(l_leg*cos(X2[i]))
      RH_x2.append(l_leg*sin(X2[i])+hip*sin(Y2[i]))
      RH_y2.append(l_leg*cos(X2[i])+hip*cos(Y2[i]))
      RA_x2.append(l_leg*sin(X2[i])+hip*sin(Y2[i])+r_leg*sin(Z2[i]))
      RA_y2.append(l_leg*cos(X2[i])+hip*cos(Y2[i])+r_leg*cos(Z2[i]))
    
x3 = []
y3 = []
z3 = []
det_three = []
LA_x3 = []
LA_y3 = []
RH_x3 = []
RH_y3 = []
RA_x3 = []
RA_y3 = []

"""
for i in range(len(det_3)):
  if(det_3[i] < 0.0001):
    if(X3[i] < 1.57 and X3[i] > -1.57):
      x3.append(X3[i])
      y3.append(Y3[i])
      z3.append(Z3[i])
      det_three.append(det_3[i])
      LA_x3.append(l_leg*sin(X3[i]))
      LA_y3.append(l_leg*cos(X3[i]))
      RH_x3.append(l_leg*sin(X3[i])+hip*sin(Y3[i]))
      RH_y3.append(l_leg*cos(X3[i])+hip*cos(Y3[i]))
      RA_x3.append(l_leg*sin(X3[i])+hip*sin(Y3[i])+r_leg*sin(Z3[i]))
      RA_y3.append(l_leg*cos(X3[i])+hip*cos(Y3[i])+r_leg*cos(Z3[i]))

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, aspect='equal', xlim=(-2,2),ylim = (-2,2))
for i in range(len(LA_x2)):
  thisx = [0, LA_x2[i], RH_x2[i], RA_x2[i]]
  thisy = [0, LA_y2[i], RH_y2[i], RA_y2[i]]
  plt.plot(thisx, thisy)
plt.show()
  

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, aspect='equal', xlim=(-2,2),ylim = (-2,2))
for i in range(len(LA_x3)):
  thisx = [0, LA_x3[i], RH_x3[i], RA_x3[i]]
  thisy = [0, LA_y3[i], RH_y3[i], RA_y3[i]]
  plt.plot(thisx, thisy)
plt.show()



fig = plt.figure()
ax = fig.gca(projection = '3d')
c = det_two
ax.scatter(x2, y2, det_two, c=c)

ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('det_two')

plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = det_three
ax.scatter(y3, z3, det_three, c=c)

ax.set_xlabel('theta_2')
ax.set_ylabel('theta_3')
ax.set_zlabel('det_three')

plt.show()
"""
five3 = []
negfive3 = []
for i in range(len(x3)):
  five3.append(5)
  negfive3.append(-5)
five2 = []
negfive2 = []
for i in range(len(x2)):
  five2.append(5)
  negfive2.append(-5)

fig = plt.figure()
plt.scatter(x2, y2)
xlabel('angle_1')
ylabel('angle_2')
plt.grid(True)
fig.suptitle('Left Hip Uncontrolled Regions')
plt.show()
fig = plt.figure()
plt.scatter(y2, z2)
xlabel('angle_2')
ylabel('angle_3')
plt.grid(True)
fig.suptitle('Left Hip Uncontrolled Regions')
plt.show()

fig = plt.figure()
plt.scatter(x2, z2)
xlabel('angle_1')
ylabel('angle_3')
plt.grid(True)
fig.suptitle('Left Hip Uncontrolled Regions')
plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = x2
fig.suptitle('Left Hip Uncontrolled Regions')
ax.scatter(x2, y2,z2, c=c)
#ax.scatter(x2,y2, -5)
#ax.scatter(x2, five2, z2)

ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')

plt.show()


fig = plt.figure()
plt.scatter(x3, z3)
fig.suptitle('Right Hip Uncontrolled Regions')
xlabel('angle_1')
ylabel('angle_3')
plt.grid(True)
plt.show()

fig = plt.figure()
plt.scatter(y3, z3)
fig.suptitle('Right Hip Uncontrolled Regions')
xlabel('angle_2')
ylabel('angle_3')
plt.grid(True)
plt.show()

fig = plt.figure()
plt.scatter(x3, y3)
fig.suptitle('Right Hip Uncontrolled Regions')
xlabel('angle_1')
ylabel('angle_2')
plt.grid(True)
plt.show()


fig = plt.figure()
ax = fig.gca(projection = '3d')
c = x3
ax.scatter(x3, y3,z3, c=c)

ax.scatter(x3, five3, z3)
ax.scatter(x3,y3,negfive3)
fig.suptitle('Right Hip Uncontrolled Regions')
ax.set_xlabel('theta_1')
ax.set_ylabel('theta_2')
ax.set_zlabel('theta_3')

plt.show()
