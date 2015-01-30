from numpy import sin, cos, cumsum, dot, zeros
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan, atan2, Matrix, simplify, sympify
from sympy.mpmath import norm
import double_pendulum_particle_setup as dp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

dp_eigenvecs = open('dp_eigenvecs.pkl', 'rb')
dp_eigenvecs_string = pickle.load(dp_eigenvecs)
dp_eigenvecs.close()

dp_eigenvecs = sympify(dp_eigenvecs_string)

first_ratio = dp_eigenvecs[0][2][0]
second_ratio = dp_eigenvecs[1][2][0]

mass_matrix = dp.mass_matrix
parameter_dict = dp.parameter_dict

mass_matrix = mass_matrix.subs(parameter_dict)
mass_matrix_inv = mass_matrix.inv()

torques = Matrix(dp.torques)

torques_dict = dict(zip([dp.one_torque, dp.two_torque], [-1.0, 1.0]))
torques_subbed = dp.torques.subs(torques_dict)

theta_dict = dict(zip([dp.theta1, dp.theta2], [0.0,0.0]))
mass_matrix_zero_subbed = mass_matrix.subs(theta_dict)
mass_matrix_zero_inv_subbed = mass_matrix_zero_subbed.inv()
mass_matrix_zero = mass_matrix.subs(theta_dict)
mass_matrix_zero_inv = mass_matrix_zero.inv()
first_ratio_num = first_ratio.subs(parameter_dict)
second_ratio_num = second_ratio.subs(parameter_dict)
first_ratio_acc = mass_matrix_zero_inv*first_ratio_num
second_ratio_acc = mass_matrix_zero_inv*second_ratio_num

vals_0 = []
vals_1 = []
for i in np.arange(1.5, 1.61, 0.001):
    d = dict(zip([dp.one_torque, dp.two_torque], [i, 1.0]))
    a = mass_matrix_zero_inv*(torques.subs(d))
    vals_0.append(a[0])
    vals_1.append(a[1])
fig = plt.figure()
e_vals_0 = []
e_vals_1 = []
e1_vals_0 = []
e1_vals_1 = []

for i in np.arange(-1.57, 1.57, 0.01):
    d = dict(zip([dp.theta1, dp.theta2], [0,i]))
    a = second_ratio_num.subs(d)
    b = first_ratio_num.subs(d)
    c = (mass_matrix_inv.subs(d))*b
    k = (mass_matrix_inv.subs(d))*a
    e_vals_0.append(k[0])
    e_vals_1.append(k[1])
    e1_vals_0.append(c[0])
    e1_vals_1.append(c[1])
plt.scatter(np.arange(1.5, 1.61, 0.001), vals_0)
plt.scatter(np.arange(1.5, 1.61, 0.001), vals_1)
ei = second_ratio_acc.subs(theta_dict)
plt.scatter(second_ratio_num.subs(theta_dict)[0], ei[0], c = 'g')
plt.scatter(second_ratio_num.subs(theta_dict)[0], ei[1], c = 'r') 

#joint_one, = plt.plot(np.arange(-1.57, 1.57, 0.05), e_vals_0, c = 'r')
#joint_two, = plt.plot(np.arange(-1.57, 1.57, 0.05), e_vals_1, c = 'b')

#xlabel('angle_2')
#ylabel('acceleration')

#plt.legend([joint_one, joint_two], ['Joint 1', 'Joint 2'])

#plt.scatter(np.arange(-1.57, 1.57, 0.05), e1_vals_0, c = 'y')
#plt.scatter(np.arange(-1.57, 1.57, 0.05), e1_vals_1, c = 'purple')



plt.show()


#for i in np.arange(-1.57, 1.57, 0.05):
#    mass_matrix.subs(
