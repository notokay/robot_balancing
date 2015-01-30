from numpy import sin, cos, cumsum, dot, zeros
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan, atan2, Matrix, simplify, sympify
from sympy.mpmath import norm
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import single_pendulum_setup as sp
import double_pendulum_particle_setup as pp
import pickle

parameter_dict = dict(zip(pp.constants, pp.numerical_constants))

global_mass_dict = dict(zip([sp.s_mass], [pp.one_mass+pp.two_mass]))

lam_com_eq = lambdify([pp.theta1, pp.theta2], pp.com.subs(parameter_dict))
lam_com_dot_eq = lambdify([pp.theta1, pp.theta2, pp.omega1, pp.omega2], pp.com_dot.subs(parameter_dict))
lam_sp_eq_torque_only = lambdify([sp.s_length, sp.s_torque], (sp.s_torque*(sp.mass_matrix.inv()))[0].subs(global_mass_dict).subs(parameter_dict))

lam_sp_com_x = lambdify([sp.s_length, sp.theta_s, sp.omega_s, sp.alpha_s], sp.com_ddot[0])
lam_sp_com_y = lambdify([sp.s_length, sp.theta_s, sp.omega_s, sp.alpha_s], sp.com_ddot[1])

lam_solved_pp_com_ddot_t1 = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y], pp.des_t1_acc_xy)
lam_solved_pp_com_ddot_t2 = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y], pp.des_t2_acc_xy)

lam_solved_pp_com_ddot_t1_euler = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc], pp.des_t1_acc_euler)

lam_solved_pp_com_ddot_t2_euler = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc], pp.des_t2_acc_euler)

lam_com_ddot_matrix = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2], pp.com_ddot_matrix)

k_p = 0.05
k_v = 0.05

def get_global_com_params(state):
    com_loc = Matrix(lam_com_eq(state[0], state[1]))
    com_x = com_loc[0]
    com_y = com_loc[1]
    com_pen_length = (com_x**2+com_y**2)**0.5
    com_pen_theta = -1*atan(com_x/com_y)
    params = dict(zip(['sp_length', 'sp_theta', 'sp_com_xy'], [com_pen_length, com_pen_theta, com_loc]))
    return params

def get_sp_des_xy_acc(params, state):
    com_vel_xy = Matrix(lam_com_dot_eq(state[0], state[1], state[2], state[3]))
    com_ang_vel = (params.get('sp_com_xy').cross(com_vel_xy))/(norm(params.get('sp_com_xy')))**2
    com_ang_vel = com_ang_vel[2]
    com_u = -k_p*params.get('sp_theta') + -k_v*com_ang_vel
    com_ang_acc = Matrix([0,0,lam_sp_eq_torque_only(params.get('sp_length'), com_u)])
    com_ang_acc = com_ang_acc[2]
    com_ang_acc_xy = Matrix([lam_sp_com_x(params.get('sp_length'), params.get('sp_theta'), com_ang_vel, com_ang_acc), lam_sp_com_y(params.get('sp_length'), params.get('sp_theta'), com_ang_vel, com_ang_acc)])
    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y', 'ang_acc'], [com_ang_acc_xy[0], com_ang_acc_xy[1], com_ang_acc]))
    return com_ang_acc_dict
def get_dp_acc(params, state):
    ddot_mat = Matrix(lam_com_ddot_matrix(params.get(pp.one_length),
                                   params.get(pp.two_length),
                                   params.get(pp.one_mass),
                                   params.get(pp.two_mass),
                                   state[0],
                                   state[1],
                                   state[2],
                                   state[3]))
    print(state)
    if(ddot_mat.rank() == 2):
    ddot_mat_pinv = ddot_mat.pinv()
    u,s,v = np.linalg.svd(ddot_mat_pinv)
    for i in s:
        if i < 1.0:
            break
        else:
            ddot_mat.row_del(2)
            acc = ddot_mat.nullspace()[0]
            acc.row_del(2)
            return acc
    ddot_mat_inv = ddot_mat.inv()
    u, s, v = np.linalg.svd(ddot_mat_inv)
    for i in range(len(s)):
        if s[i] < 1.0:
            s[i] = 0.0
        else:
            s[i] = 1.0/s[i]
    ddot_mat_rec = Matrix(np.dot(v, np.dot(np.diag(s), u.T)))
    ddot_mat_rec.row_del(2)
    acc = (ddot_mat_rec.nullspace()[0])
    acc.row_del(2)
    return acc

t1_des_xy = []
t2_des_xy = []
t1_des_euler = []
t2_des_euler = []
t1_nullspace = []
t2_nullspace = []
for i in np.arange(-1., 1.,0.11):
    for j in np.arange(-1., 1., 0.11):
        state = [i,j,np.sign(i)*0.1,np.sign(j)*0.1]
#        params = get_global_com_params(state)
#        sp_des_acc = get_sp_des_xy_acc(params, state)
        dp_des_acc = get_dp_acc(parameter_dict, state)
#        t1_des.append([i, j, lam_solved_pp_com_ddot_t1(1.0, 2.0, 1.0, 2.0, i, j, np.sign(i)*0.1, np.sign(j)*0.1, sp_des_acc.get('ang_acc_x'), sp_des_acc.get('ang_acc_y')) ])
#        t2_des.append([i, j, lam_solved_pp_com_ddot_t2(1.0, 2.0, 1.0, 2.0, i, j, np.sign(i)*0.1, np.sign(j)*0.1, sp_des_acc.get('ang_acc_x'), sp_des_acc.get('ang_acc_y')) ])
#        t1_des.append( [i, j, lam_solved_pp_com_ddot_t1_euler(1.0, 2.0, 1.0, 2.0, i, j, np.sign(i)*0.1, np.sign(j)*0.1, sp_des_acc.get('ang_acc_x'), sp_des_acc.get('ang_acc')) ])
#        t2_des.append([i, j, lam_solved_pp_com_ddot_t2_euler(1.0, 2.0, 1.0, 2.0, i, j, np.sign(i)*0.1, np.sign(j)*0.1, sp_des_acc.get('ang_acc_x'), sp_des_acc.get('ang_acc')) ])
        t1_nullspace.append([i, j, dp_des_acc[0]])
        t2_nullspace.append([i, j, dp_des_acc[1]])


fig = plt.figure()



ax = fig.add_subplot(111, projection='3d')

t1_nullspace = np.array(np.array(t1_nullspace), dtype = float)
t2_nullspace = np.array(np.array(t2_nullspace), dtype = float)

#ax.scatter(t1_des[:,0], t1_des[:,1], t1_des[:,2])
ax.plot_trisurf(t1_nullspace[:,0].flatten(), t1_nullspace[:,1].flatten(), t1_nullspace[:,2].flatten(), color = 'r')

#ax.plot_trisurf(t2_des[:,0].flatten(), t2_des[:,1].flatten(), t2_des[:,2].flatten(), color = 'g')

plt.show()
