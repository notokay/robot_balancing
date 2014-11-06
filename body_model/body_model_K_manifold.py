from sympy import symbols, simplify, lambdify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, sin, cos, pi, zeros, dot, eye
from numpy.linalg import inv
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.physics.vector import init_vprinting, vlatex
from math import fabs
init_vprinting()
import pickle
from mpl_toolkits.mplot3d import Axes3D


# Control
# =======

inputK = open('bm_LQR_K_useful.pkl','rb')
inputa1 = open('bm_angle_one_useful_1.pkl','rb')
inputa2 = open('bm_angle_two_useful_1.pkl','rb')
inputa3 = open('bm_angle_three_useful_1.pkl','rb')
inputa4 = open('bm_angle_four_useful_1.pkl','rb')

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

theta1_theta1 = []
theta1_theta2 = []
theta1_theta3 = []
theta1_theta4 = []
theta1_omega1 = []
theta1_omega2 = []
theta1_omega3 = []
theta1_omega4 = []

theta2_theta1 = []
theta2_theta2 = []
theta2_theta3 = []
theta2_theta4 = []
theta2_omega1 = []
theta2_omega2 = []
theta2_omega3 = []
theta2_omega4 = []


theta3_theta1 = []
theta3_theta2 = []
theta3_theta3 = []
theta3_theta4 = []
theta3_omega1 = []
theta3_omega2 = []
theta3_omega3 = []
theta3_omega4 = []


theta4_theta1 = []
theta4_theta2 = []
theta4_theta3 = []
theta4_theta4 = []
theta4_omega1 = []
theta4_omega2 = []
theta4_omega3 = []
theta4_omega4 = []

for element in K:
  theta2_theta1.append(element[1][0])
  theta2_theta2.append(element[1][1])
  theta2_theta3.append(element[1][2])
  theta2_theta4.append(element[1][3])  
  theta2_omega1.append(element[1][4])
  theta2_omega2.append(element[1][5])
  theta2_omega3.append(element[1][6])
  theta2_omega4.append(element[1][7])

  theta3_theta1.append(element[2][0])
  theta3_theta2.append(element[2][1])
  theta3_theta3.append(element[2][2])
  theta3_theta4.append(element[2][3])  
  theta3_omega1.append(element[2][4])
  theta3_omega2.append(element[2][5])
  theta3_omega3.append(element[2][6])
  theta3_omega4.append(element[2][7])

  theta4_theta1.append(element[3][0])
  theta4_theta2.append(element[3][1])
  theta4_theta3.append(element[3][2])
  theta4_theta4.append(element[3][3])  
  theta4_omega1.append(element[3][4])
  theta4_omega2.append(element[3][5])
  theta4_omega3.append(element[3][6])
  theta4_omega4.append(element[3][7])

A = []
b1 = []

for t1, t2, t3, t4  in zip(a1, a2, a3, a4):
  A.append([1, t1, t1**2, t2, t2**2, t3, t3**2, t4, t4**2, 0.000001, 0.000001**2, 0.000001, 0.000001**2, 0.000001, 0.000001**2, 0.000001, 0.000001**2,])

t2t1_eq = np.linalg.lstsq(A, theta2_theta1)[0]
t2t2_eq = np.linalg.lstsq(A, theta2_theta2)[0]
t2t3_eq = np.linalg.lstsq(A, theta2_theta3)[0]
t2t4_eq = np.linalg.lstsq(A, theta2_theta4)[0]
t2o1_eq = np.linalg.lstsq(A, theta2_omega1)[0]
t2o2_eq = np.linalg.lstsq(A, theta2_omega2)[0]
t2o3_eq = np.linalg.lstsq(A, theta2_omega3)[0]
t2o4_eq = np.linalg.lstsq(A, theta2_omega4)[0]

t3t1_eq = np.linalg.lstsq(A, theta3_theta1)[0]
t3t2_eq = np.linalg.lstsq(A, theta3_theta2)[0]
t3t3_eq = np.linalg.lstsq(A, theta3_theta3)[0]
t3t4_eq = np.linalg.lstsq(A, theta3_theta4)[0]
t3o1_eq = np.linalg.lstsq(A, theta3_omega1)[0]
t3o2_eq = np.linalg.lstsq(A, theta3_omega2)[0]
t3o3_eq = np.linalg.lstsq(A, theta3_omega3)[0]
t3o4_eq = np.linalg.lstsq(A, theta3_omega4)[0]

t4t1_eq = np.linalg.lstsq(A, theta4_theta1)[0]
t4t2_eq = np.linalg.lstsq(A, theta4_theta2)[0]
t4t3_eq = np.linalg.lstsq(A, theta4_theta3)[0]
t4t4_eq = np.linalg.lstsq(A, theta4_theta4)[0]
t4o1_eq = np.linalg.lstsq(A, theta4_omega1)[0]
t4o2_eq = np.linalg.lstsq(A, theta4_omega2)[0]
t4o3_eq = np.linalg.lstsq(A, theta4_omega3)[0]
t4o4_eq = np.linalg.lstsq(A, theta4_omega4)[0]

gain_coefs = array([[t2t1_eq, t2t2_eq, t2t3_eq, t2t4_eq, t2o1_eq, t2o2_eq, t2o3_eq, t2o4_eq], [t3t1_eq, t3t2_eq, t3t3_eq, t3t4_eq, t3o1_eq, t3o2_eq, t3o3_eq, t3o4_eq], [t4t1_eq, t4t2_eq, t4t3_eq, t4t4_eq, t4o1_eq, t4o2_eq, t4o3_eq, t4o4_eq]])

outputG = open('body_model_gain_coefs.pkl', 'wb')
pickle.dump(gain_coefs, outputG)

outputG.close()

#test_gain = []

#for t1, t2, t3, t4 in zip(a1, a2, a3, a4):
#  test_gain.append(t2t3_eq[0] + t2t3_eq[1]*t1 + t2t3_eq[2]*t1**2 + t2t3_eq[3]*t2 + t2t3_eq[4]*t2**2 + t2t3_eq[5]*t3 + t2t3_eq[6]*t3**2 + t2t3_eq[7]*t4 + t2t3_eq[8]*t4**2)

#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#ax.scatter(a2,a3, test_gain)
#ax.set_xlabel("theta_2")
#ax.set_ylabel("theta_3")
#ax.set_zlabel("gain")
#plt.show()
