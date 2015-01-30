from numpy import sin, cos, cumsum, dot, zeros
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan, atan2, Matrix, simplify
from sympy.mpmath import norm
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
import math
import numpy as np
import single_pendulum_setup as sp
import double_pendulum_particle_setup as dp
import triple_pendulum_setup as tp

# Specify Numerical Quantities
# ============================
right_hand_side = generate_ode_function(tp.mass_matrix_full, tp.forcing_vector_full, 
                                        tp.constants, tp.coordinates,
                                        tp.speeds, tp.specified)
initial_coordinates = [0.01, -0.1, 0.01]

initial_speeds = [-0.0,-0.0, -0.0]
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([0.5,
                             0.5,
                             0.75,
                             0.75,
                             1.0,
                             1.0,
                             9.81])

parameter_dict = dict(zip(tp.constants, numerical_constants))


args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}

# Simulate
# ========

torque_vector = []
time_vector = []
x_vector = []

frames_per_sec = 60
final_time = 2.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

right_hand_side(x0, 0.0, args)

gravity_dict = dict(zip([sp.g, dp.g, tp.g], [9.81,9.81, 9.81]))
global_mass_dict = dict(zip([sp.s_mass], [tp.a_mass+tp.b_mass+tp.c_mass]))
sp_forcing = -1*simplify(sp.forcing - sp.s_torque) + sp.s_torque
sp_eq = (sp.mass_matrix.inv()*sp_forcing)[0].subs(gravity_dict)
lam_sp_eq = lambdify([sp.s_length, sp.theta_s, sp.s_torque], sp_eq.subs(global_mass_dict).subs(parameter_dict))
lam_sp_eq_torque_only = lambdify([sp.s_length, sp.s_torque], (sp.s_torque*(sp.mass_matrix.inv()))[0].subs(global_mass_dict).subs(parameter_dict) )

lam_com_eq = lambdify((tp.theta_a, tp.theta_b, tp.theta_c), tp.com.subs(parameter_dict))

lam_com_dot_eq = lambdify([tp.theta_a, tp.theta_b, tp.theta_c, 
                           tp.omega_a, tp.omega_b, tp.omega_c], 
                          tp.com_dot.subs(parameter_dict))

lam_solved_dp_com_ddot_t1 = lambdify([dp.one_length, dp.two_length, dp.one_mass, dp.two_mass, dp.theta1, dp.theta2, dp.omega1, dp.omega2, dp.des_com_ang_acc_x, dp.des_com_ang_acc_y], dp.des_t1_acc)
lam_solved_dp_com_ddot_t2 = lambdify([dp.one_length, dp.two_length, dp.one_mass, dp.two_mass, dp.theta1, dp.theta2, dp.omega1, dp.omega2, dp.des_com_ang_acc_x, dp.des_com_ang_acc_y], dp.des_t2_acc)

lam_bc_com = lambdify([tp.theta_b, tp.theta_c], tp.com_bc_in_a.subs(parameter_dict))
lam_bc_com_inertial = lambdify([tp.theta_a, tp.theta_b, tp.theta_c], tp.com_bc.subs(parameter_dict))
lam_bc_com_dot = lambdify([tp.theta_b, tp.theta_c, tp.omega_b, tp.omega_c], tp.com_bc_dot_in_a.subs(parameter_dict))

lam_dp_inverse_dynamics = lambdify([dp.one_length, dp.two_length, dp.one_mass, dp.two_mass, dp.theta1, dp.theta2, dp.omega1, dp.omega2, dp.alpha1, dp.alpha2], dp.inverse_dynamics.subs(gravity_dict))

k_p = .0
k_v = .0

def get_sp_des_xy_acc(params, state): #returns desired com x and y acceleration based off of PD control of the center of mass pendulum
    global k_p, k_v
    com_vel_xy = Matrix(lam_com_dot_eq(state[0], state[1], state[2], state[3], state[4], state[5]))
    com_ang_vel = (params.get('sp_com_xy').cross(com_vel_xy))/(norm(params.get('sp_com_xy')))**2
    com_ang_vel = com_ang_vel[2]
    com_u = -k_p*params.get('sp_theta') + -k_v*com_ang_vel #pd torque control

#    com_ang_acc = Matrix([0,0, lam_sp_eq(params.get('sp_length'), params.get('sp_theta'), com_u)]) #angular acceleration resulting from pd torque control
    com_ang_acc = Matrix([0,0, lam_sp_eq_torque_only(params.get('sp_length'), com_u)])
    com_ang_acc_xy = (com_ang_acc.cross(params.get('sp_com_xy')))
    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y'], [com_ang_acc_xy[0], com_ang_acc_xy[1]]))
#    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y'], [com_ang_acc_xy[0], 0.0]))
    return com_ang_acc_dict

def get_dp_angular_from_com_xy_acc(one_params, two_params, desired_xy_acc):
    b = two_params.get('angle')
    q = math.sin(b)
    if q == 0.0:
        return dict(zip(['theta1_acc', 'theta2_acc'], [0.,0.]))
    theta_1_acc = lam_solved_dp_com_ddot_t1(one_params.get('length'), 
                                            two_params.get('length'), 
                                            one_params.get('mass'), 
                                            two_params.get('mass'), 
                                            one_params.get('angle'), 
                                            two_params.get('angle'), 
                                            one_params.get('ang_vel'), 
                                            two_params.get('ang_vel'), 
                                            desired_xy_acc.get('ang_acc_x'), 
                                            desired_xy_acc.get('ang_acc_y'))
    theta_2_acc = lam_solved_dp_com_ddot_t2(one_params.get('length'), 
                                            two_params.get('length'), 
                                            one_params.get('mass'), 
                                            two_params.get('mass'), 
                                            one_params.get('angle'), 
                                            two_params.get('angle'), 
                                            one_params.get('ang_vel'), 
                                            two_params.get('ang_vel'), 
                                            desired_xy_acc.get('ang_acc_x'), 
                                            desired_xy_acc.get('ang_acc_y'))

    desired_dp_theta_acc_dict = dict(zip(['theta1_acc', 'theta2_acc'], [theta_1_acc, theta_2_acc]))
    return desired_dp_theta_acc_dict

def get_global_com_params(state): #yields com pendulum length, angle, and x-y coordinate of endpoint
    com_loc = Matrix(lam_com_eq(state[0], state[1], state[2]))
    com_x = com_loc[0]
    com_y = com_loc[1]
    com_pen_length = (com_x**2 + com_y**2)**0.5
    com_pen_theta = -1*atan(com_x/com_y)
    params = dict(zip(['sp_length', 'sp_theta', 'sp_com_xy'], [com_pen_length, com_pen_theta, com_loc]))
    return params

def get_dp_bc_params(state):
    dp_bc_xy = Matrix(lam_bc_com(state[1], state[2]))
    dp_bc_xy_inertial = Matrix(lam_bc_com_inertial(state[0], state[1], state[2]))
    dp_bc_length = (dp_bc_xy[0]**2 + dp_bc_xy[1]**2)**0.5
    dp_bc_angle = -atan(dp_bc_xy[0]/dp_bc_xy[1]) #angle with respect to angle one
    dp_bc_ang_vel = (dp_bc_xy.cross(Matrix(lam_bc_com_dot(state[1], state[2], state[4], state[5]))))/norm(dp_bc_xy)**2
    dp_bc_params_dict = dict(zip(['bc_length', 'bc_theta', 'bc_ang_vel', 'bc_loc'], [dp_bc_length, dp_bc_angle, dp_bc_ang_vel[2], dp_bc_xy_inertial])) 
    return dp_bc_params_dict

def dp_inverse_dynamics(one_params, two_params, desired_angular_acc):
    required_torques = Matrix(lam_dp_inverse_dynamics(one_params.get('length'), 
                                               two_params.get('length'), 
                                               one_params.get('mass'), 
                                               two_params.get('mass'), 
                                               one_params.get('angle'), 
                                               two_params.get('angle'), 
                                               one_params.get('ang_vel'), 
                                               two_params.get('ang_vel'), 
                                               desired_angular_acc.get('theta1_acc'), 
                                               desired_angular_acc.get('theta2_acc')))
    
    return dict(zip(['one_torque', 'two_torque'], required_torques))

def controller(x, t):
    global torque_vector, time_vector, x_vector
    returnval = [0., 0., 0.]
    global_sp_params = get_global_com_params(x) #convert from triple pendulum to single pendulum
    desired_global_sp_acc_xy = get_sp_des_xy_acc(global_sp_params, x) #calculates desired single pendulum acceleration
    dp_bc_params = get_dp_bc_params(x)
    dp_two_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'], 
                             [dp_bc_params.get('bc_length'),
                              parameter_dict.get(tp.b_mass) + parameter_dict.get(tp.c_mass),
                              x[1] + x[2], 
                              dp_bc_params.get('bc_ang_vel')])) #params for link 2 of dp1 model
    dp_one_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'], 
                             [parameter_dict.get(tp.a_length),
                              parameter_dict.get(tp.a_mass), 
                              x[0], 
                              x[3]])) #params for link 1 of dp1 model
    dp_angular = get_dp_angular_from_com_xy_acc(dp_one_params, dp_two_params, desired_global_sp_acc_xy) #yields desired a1 and a2 for dp1 based on desired single pendulum acceleration
    dp_desired_torques = dp_inverse_dynamics(dp_one_params, dp_two_params, dp_angular) #torque_a set to that required to produce desired dp acceleration by inverse dynamics
    returnval[0] = dp_desired_torques.get('one_torque')
    
    dp2_com_ang_acc = (dp_bc_params.get('bc_loc').cross(Matrix([0,0,dp_angular.get('theta2_acc')])))/(norm(dp_bc_params.get('bc_loc'))**2)
    dp2_com_ang_acc = dict(zip(['ang_acc_x', 'ang_acc_y'], [dp2_com_ang_acc[0], dp2_com_ang_acc[1]]))
    dp_one_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'],
                             [parameter_dict.get(tp.b_length),
                              parameter_dict.get(tp.b_mass),
                              x[0] + x[1],
                              x[3] + x[4]]))
    dp_two_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'],
                             [parameter_dict.get(tp.c_length),
                              parameter_dict.get(tp.c_mass),
                              x[2],
                              x[5]]))
                              
    dp_angular = get_dp_angular_from_com_xy_acc(dp_one_params, dp_two_params, dp2_com_ang_acc)
    dp_desired_torques = dp_inverse_dynamics(dp_one_params, dp_two_params, dp_angular)
    returnval[1] = dp_desired_torques.get('one_torque')
    returnval[2] = dp_desired_torques.get('two_torque')
    
    if(returnval[0] > 10.0):
        returnval[0] = 10.0
    if(returnval[0] < -10.0):
        returnval[0] = -10.0
    if(returnval[1] > 10.0):
        returnval[1] = 10.0
    if(returnval[1] < -10.0):
        returnval[1] = -10.0
    if(returnval[2] > 10.0):
        returnval[2] = 10.0
    if(returnval[2] < -10.0):
        returnval[2] = -10.0
    torque_vector.append(returnval)
    time_vector.append(t)
    x_vector.append(x)
    return returnval

def test_control(x,t):
    return [1.0, 2.0, 3.0]

args['specified'] = controller

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

x1 = -1*numerical_constants[0]*np.sin(y[:,0])
y1 = numerical_constants[0]*np.cos(y[:,0])

x2 = x1 + -1*numerical_constants[2]*np.sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[2]*np.cos(y[:,0] + y[:,1])

x3 = x2 + -1*numerical_constants[4]*np.sin(y[:,0] + y[:,1] + y[:,2])
y3 = y2 + numerical_constants[4]*np.cos(y[:,0] + y[:,1] + y[:,2])

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]

    line.set_data(thisx, thisy)
    time_text.set_text('time = %.1f' % (i*dt))
    return line, time_text

# choose the interval based on dt and the time to animate one step
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()

