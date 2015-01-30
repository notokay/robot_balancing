from numpy import sin, cos, cumsum, dot, zeros
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan, atan2, Matrix, simplify, sympify
from sympy.mpmath import norm
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
import math
import numpy as np
import single_pendulum_setup as sp
import double_pendulum_particle_setup as pp
import double_pendulum_rb_setup as rb
import pickle
print("done importing!")
# Specify Numerical Quantities
# ============================
file = open('args_ode_funcs.pkl', 'rb')
args_ode_funcs = pickle.load(file)
file.close()
args_ode_funcs = sympify(args_ode_funcs)
right_hand_side = generate_ode_function(args_ode_funcs[0],args_ode_funcs[1],args_ode_funcs[2],args_ode_funcs[3],args_ode_funcs[4],args_ode_funcs[5])
initial_coordinates = [0.01, 0.01, 0.01]

initial_speeds = [0.0, 0.0, 0.0]
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([0.5,
                             0.5,
                             0.75,
                             0.75,
                             1.0,
                             1.0,
                             9.81])

parameter_dict = dict(zip(args_ode_funcs[2], numerical_constants))
tp_const = args_ode_funcs[2]
tp_a_length = tp_const[0]
tp_a_mass = tp_const[1]
tp_b_length = tp_const[2]
tp_b_mass = tp_const[3]
tp_c_length = tp_const[4]
tp_c_mass = tp_const[5]

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}

# Simulate
# ========

torque_vector = []
time_vector = []
x_vector = []
com_vector = []
des_acc = []

frames_per_sec = 60
final_time = 0.5

t = linspace(0.0, final_time, final_time * frames_per_sec)

right_hand_side(x0, 0.0, args)

gravity_dict = dict(zip([sp.g, pp.g, rb.g], [9.81,9.81, 9.81]))
global_mass_dict = dict(zip([sp.s_mass], [tp_a_mass+tp_b_mass+tp_c_mass]))
sp_forcing = -1*simplify(sp.forcing - sp.s_torque) + sp.s_torque
sp_eq = (sp.mass_matrix.inv()*sp_forcing)[0].subs(gravity_dict)
lam_sp_eq = lambdify([sp.s_length, sp.theta_s, sp.s_torque], sp_eq.subs(global_mass_dict).subs(parameter_dict))
lam_sp_eq_torque_only = lambdify([sp.s_length, sp.s_torque], (sp.s_torque*(sp.mass_matrix.inv()))[0].subs(global_mass_dict).subs(parameter_dict) )
lam_sp_com_x = lambdify([sp.s_length, sp.theta_s, sp.omega_s, sp.alpha_s], sp.com_ddot[0])
lam_sp_com_y = lambdify([sp.s_length, sp.theta_s, sp.omega_s, sp.alpha_s], sp.com_ddot[1])

file = open('com_eq.pkl', 'rb')
com_eq = pickle.load(file)
file.close()
lam_com_eq = lambdify(sympify(com_eq[1]), sympify(com_eq[0]))

file = open('com_dot_eq.pkl', 'rb')
com_dot_eq = pickle.load(file)
file.close()
lam_com_dot_eq = lambdify(sympify(com_dot_eq[1]), sympify(com_dot_eq[0]))

file = open('a_com.pkl', 'rb')
a_com = pickle.load(file)
file.close()
lam_a_com = lambdify(sympify(a_com[1]), sympify(a_com[0]))

file = open('a_com_dot.pkl', 'rb')
a_com_dot = pickle.load(file)
file.close()
lam_a_com_dot = lambdify(sympify(a_com_dot[1]), sympify(a_com_dot[0]))

lam_solved_pp_com_ddot_t1 = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y], pp.des_t1_acc)
lam_solved_pp_com_ddot_t2 = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y], pp.des_t2_acc)

lam_solved_rb_com_ddot_t1 = lambdify([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_com_x_dot, rb.one_com_y_dot, rb.two_com_x_dot, rb.two_com_y_dot, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.des_com_ang_acc_x, rb.des_com_ang_acc_y], rb.des_t1_acc)
lam_solved_rb_com_ddot_t2 = lambdify([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_com_x_dot, rb.one_com_y_dot, rb.two_com_x_dot, rb.two_com_y_dot, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.des_com_ang_acc_x, rb.des_com_ang_acc_y], rb.des_t2_acc)

file = open('bc_com.pkl', 'rb')
bc_com = pickle.load(file)
file.close()
lam_bc_com = lambdify(sympify(bc_com[1]), sympify(bc_com[0]))

file = open('bc_com_inertial.pkl', 'rb')
bc_com_inertial = pickle.load(file)
file.close()
lam_bc_com_inertial = lambdify(sympify(bc_com_inertial[1]), sympify(bc_com_inertial[0]))

file = open('bc_com_dot.pkl', 'rb')
bc_com_dot = pickle.load(file)
file.close()
lam_bc_com_dot = lambdify(sympify(bc_com_dot[1]), sympify(bc_com_dot[0]))

lam_pp_inverse_dynamics = lambdify([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.alpha1, pp.alpha2], pp.inverse_dynamics.subs(gravity_dict))

lam_rb_inverse_dynamics = lambdify([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_com_x_dot, rb.one_com_y_dot, rb.two_com_x_dot, rb.two_com_y_dot, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.alpha1, rb.alpha2], rb.inverse_dynamics.subs(gravity_dict))

k_p = 0.05
k_v = 0.05

def get_sp_des_xy_acc(params, state):
    com_vel_xy = Matrix(lam_com_dot_eq(state[0], state[1], state[2], state[3], state[4], state[5]))
    com_ang_vel = (params.get('sp_com_xy').cross(com_vel_xy))/(norm(params.get('sp_com_xy')))**2
    com_ang_vel = com_ang_vel[2]
    com_u = -k_p*params.get('sp_theta') + -k_v*com_ang_vel #pd torque control
#
    com_ang_acc = Matrix([0,0, lam_sp_eq_torque_only(params.get('sp_length'), com_u)])
    com_ang_acc = com_ang_acc[2]
#    
    com_ang_acc_xy = Matrix([lam_sp_com_x(params.get('sp_length'), params.get('sp_theta'), com_ang_vel, com_ang_acc), lam_sp_com_y(params.get('sp_length'), params.get('sp_theta'), com_ang_vel, com_ang_acc)])
    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y', 'ang_acc'], [com_ang_acc_xy[0], com_ang_acc_xy[1], com_ang_acc]))
    return com_ang_acc_dict

def get_sp_des_xy_acc_old(params, state): #returns desired com x and y acceleration based off of PD control of the center of mass pendulum
    global k_p, k_v
    com_vel_xy = Matrix(lam_com_dot_eq(state[0], state[1], state[2], state[3], state[4], state[5]))
    com_ang_vel = (params.get('sp_com_xy').cross(com_vel_xy))/(norm(params.get('sp_com_xy')))**2
    com_ang_vel = com_ang_vel[2]
    com_u = -k_p*params.get('sp_theta') + -k_v*com_ang_vel #pd torque control

#    com_ang_acc = Matrix([0,0, lam_sp_eq(params.get('sp_length'), params.get('sp_theta'), com_u)]) #angular acceleration resulting from pd torque control
    com_ang_acc = Matrix([0,0, lam_sp_eq_torque_only(params.get('sp_length'), com_u)])
    com_ang_acc_xy = (com_ang_acc.cross(params.get('sp_com_xy')))
    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y'], [com_ang_acc_xy[0], com_ang_acc_xy[1]]))
#    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y'], [0.0, 0.0]))
#    com_ang_acc_dict = dict(zip(['ang_acc_x', 'ang_acc_y'], [com_ang_acc_xy[0], 0.0]))
    return com_ang_acc_dict

def get_pp_angular_from_com_xy_acc(one_params, two_params, desired_xy_acc):
    b = two_params.get('angle')
    q = math.sin(b)
    if q == 0.0:
        return dict(zip(['theta1_acc', 'theta2_acc'], [0.,0.]))
    theta_1_acc = lam_solved_pp_com_ddot_t1(one_params.get('length'), 
                                            two_params.get('length'), 
                                            one_params.get('mass'), 
                                            two_params.get('mass'), 
                                            one_params.get('angle'), 
                                            two_params.get('angle'), 
                                            one_params.get('ang_vel'), 
                                            two_params.get('ang_vel'), 
                                            desired_xy_acc.get('ang_acc_x'), 
                                            desired_xy_acc.get('ang_acc_y'))
    theta_2_acc = lam_solved_pp_com_ddot_t2(one_params.get('length'), 
                                            two_params.get('length'), 
                                            one_params.get('mass'), 
                                            two_params.get('mass'), 
                                            one_params.get('angle'), 
                                            two_params.get('angle'), 
                                            one_params.get('ang_vel'), 
                                            two_params.get('ang_vel'), 
                                            desired_xy_acc.get('ang_acc_x'), 
                                            desired_xy_acc.get('ang_acc_y'))

    desired_pp_theta_acc_dict = dict(zip(['theta1_acc', 'theta2_acc'], [theta_1_acc, theta_2_acc]))
    return desired_pp_theta_acc_dict

def get_rb_angular_from_com_xy_acc(one_params, two_params, desired_xy_acc):
    b = one_params.get('angle')
    q = math.sin(b)
    if q == 0.0:
        return dict(zip(['theta1_acc', 'theta2_acc'], [0., 0.]))
    theta_1_acc = lam_solved_rb_com_ddot_t1(one_params.get('length'),
                                            one_params.get('com_x'),
                                            one_params.get('com_y'),
                                            two_params.get('com_x'),
                                            two_params.get('com_y'),
                                            one_params.get('com_x_dot'),
                                            one_params.get('com_y_dot'),
                                            two_params.get('com_x_dot'),
                                            two_params.get('com_y_dot'),
                                            one_params.get('mass'),
                                            two_params.get('mass'),
                                            one_params.get('angle'),
                                            two_params.get('angle'),
                                            one_params.get('ang_vel'),
                                            two_params.get('ang_vel'),
                                            desired_xy_acc.get('ang_acc_x'),
                                            desired_xy_acc.get('ang_acc_y'))

    theta_2_acc = lam_solved_rb_com_ddot_t2(one_params.get('length'),
                                            one_params.get('com_x'),
                                            one_params.get('com_y'),
                                            two_params.get('com_x'),
                                            two_params.get('com_y'),
                                            one_params.get('com_x_dot'),
                                            one_params.get('com_y_dot'),
                                            two_params.get('com_x_dot'),
                                            two_params.get('com_y_dot'),
                                            one_params.get('mass'),
                                            two_params.get('mass'),
                                            one_params.get('angle'),
                                            two_params.get('angle'),
                                            one_params.get('ang_vel'),
                                            two_params.get('ang_vel'),
                                            desired_xy_acc.get('ang_acc_x'),
                                            desired_xy_acc.get('ang_acc_y'))
    desired_rb_theta_acc_dict = dict(zip(['theta1_acc', 'theta2_acc'], [theta_1_acc, theta_2_acc]))
    return desired_rb_theta_acc_dict

def get_global_com_params(state): #yields com pendulum length, angle, and x-y coordinate of endpoint
    com_loc = Matrix(lam_com_eq(state[0], state[1], state[2]))
    com_x = com_loc[0]
    com_y = com_loc[1]
    com_pen_length = (com_x**2 + com_y**2)**0.5
    com_pen_theta = -1*atan(com_x/com_y)
    params = dict(zip(['sp_length', 'sp_theta', 'sp_com_xy'], [com_pen_length, com_pen_theta, com_loc]))
    return params
def get_global_com_pos(state):
    com_loc = Matrix(lam_com_eq(state[0], state[1], state[2]))
    return com_loc

def get_dp_bc_params(state):
    dp_bc_xy = Matrix(lam_bc_com(state[1], state[2]))
    dp_bc_xy_inertial = Matrix(lam_bc_com_inertial(state[0], state[1], state[2]))
    dp_bc_length = (dp_bc_xy[0]**2 + dp_bc_xy[1]**2)**0.5
    dp_bc_angle = -atan(dp_bc_xy[0]/dp_bc_xy[1]) #angle with respect to angle one
    dp_bc_vel_xy = Matrix(lam_bc_com_dot(state[1], state[2], state[4], state[5]))
    dp_bc_ang_vel = (dp_bc_xy.cross(dp_bc_vel_xy))/norm(dp_bc_xy)**2
    dp_bc_params_dict = dict(zip(['bc_length', 'bc_theta', 'bc_ang_vel', 'bc_loc', 'bc_vel_xy'], [dp_bc_length, dp_bc_angle, dp_bc_ang_vel[2], dp_bc_xy_inertial, dp_bc_vel_xy])) 
    return dp_bc_params_dict

def get_dp_bc_com_loc(state):
    dp_bc_xy_inertial = Matrix(lam_bc_com_inertial(state[0], state[1], state[2]))
    return dp_bc_xy_inertial

def pp_inverse_dynamics(one_params, two_params, desired_angular_acc):
    required_torques = Matrix(lam_pp_inverse_dynamics(one_params.get('length'), 
                                               two_params.get('length'), 
                                               one_params.get('mass'), 
                                               two_params.get('mass'), 
                                               one_params.get('angle'), 
                                               two_params.get('angle'), 
                                               one_params.get('ang_vel'), 
                                               two_params.get('ang_vel'), 
                                               desired_angular_acc.get('theta1_acc'), 
                                               desired_angular_acc.get('theta2_acc')))
    required_torques = required_torques
    return dict(zip(['one_torque', 'two_torque'], required_torques))

def rb_inverse_dynamics(one_params, two_params, desired_angular_acc):
    required_torques = Matrix(lam_rb_inverse_dynamics(one_params.get('length'),
                                                      one_params.get('com_x'),
                                                      one_params.get('com_y'),
                                                      two_params.get('com_x'),
                                                      two_params.get('com_y'),
                                                      one_params.get('com_x_dot'),
                                                      one_params.get('com_y_dot'),
                                                      two_params.get('com_x_dot'),
                                                      two_params.get('com_y_dot'),
                                                      one_params.get('mass'), 
                                                      two_params.get('mass'), 
                                                      one_params.get('angle'), 
                                                      two_params.get('angle'), 
                                                      one_params.get('ang_vel'), 
                                                      two_params.get('ang_vel'), 
                                                      desired_angular_acc.get('theta1_acc'), 
                                                      desired_angular_acc.get('theta2_acc')))
    required_torques = required_torques
    return dict(zip(['one_torque', 'two_torque'], required_torques))

def controller(x, t):
    global torque_vector, time_vector, x_vector, com_vector
    x_vector.append(x)
#    print(x)
    returnval = [0., 0., 0.]
    global_sp_params = get_global_com_params(x) #convert from triple pendulum to single pendulum
    com_vector.append(global_sp_params.get('sp_com_xy'))
    desired_global_sp_acc_xy = get_sp_des_xy_acc(global_sp_params, x) #calculates desired single pendulum acceleration
    dp_bc_params = get_dp_bc_params(x)
    dp_two_params = dict(zip(['length', 'com_x', 'com_y', 'com_x_dot', 'com_y_dot', 'mass', 'angle', 'ang_vel'], 
                             [dp_bc_params.get('bc_length'),
                              dp_bc_params.get('bc_loc')[0],
                              dp_bc_params.get('bc_loc')[1],
                              dp_bc_params.get('bc_vel_xy')[0],
                              dp_bc_params.get('bc_vel_xy')[1],
                              parameter_dict.get(tp_b_mass) + parameter_dict.get(tp_c_mass),
                              x[1]+x[2], 
                              dp_bc_params.get('bc_ang_vel')])) #params for link 2 of dp1 model
    com_a = Matrix(lam_a_com(x[0]))
    com_a_dot = Matrix(lam_a_com_dot(x[0], x[3]))
    dp_one_params = dict(zip(['length', 'com_x', 'com_y', 'com_x_dot', 'com_y_dot', 'mass', 'angle', 'ang_vel'], 
                             [parameter_dict.get(tp_a_length),
                              com_a[0],
                              com_a[1],
                              com_a_dot[0],
                              com_a_dot[1],
                              parameter_dict.get(tp_a_mass), 
                              x[0], 
                              x[3]])) #params for link 1 of dp1 model
    rb_angular = get_pp_angular_from_com_xy_acc(dp_one_params, dp_two_params, desired_global_sp_acc_xy) #yields desired a1 and a2 for dp1 based on desired single pendulum acceleration
    des_acc.append([rb_angular.get('theta1_acc'), rb_angular.get('theta2_acc')])
    print([rb_angular.get('theta1_acc'), rb_angular.get('theta2_acc')])
    rb_desired_torques = rb_inverse_dynamics(dp_one_params, dp_two_params, rb_angular) #torque_a set to that required to produce desired dp acceleration by inverse dynamics
    returnval[0] = rb_desired_torques.get('one_torque')
    
    dp2_com_ang_acc = (dp_bc_params.get('bc_loc').cross(Matrix([0,0, rb_angular.get('theta2_acc')])))/(norm(dp_bc_params.get('bc_loc'))**2)
    dp2_com_ang_acc = dict(zip(['ang_acc_x', 'ang_acc_y'], [dp2_com_ang_acc[0], dp2_com_ang_acc[1]]))
    dp_one_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'],
                             [parameter_dict.get(tp_b_length),
                              parameter_dict.get(tp_b_mass),
                              x[0] + x[1],
                              x[4]]))
    dp_two_params = dict(zip(['length', 'mass', 'angle', 'ang_vel'],
                             [parameter_dict.get(tp_c_length),
                              parameter_dict.get(tp_c_mass),
                              x[2],
                              x[5]]))
                              
    pp_angular = get_pp_angular_from_com_xy_acc(dp_one_params, dp_two_params, dp2_com_ang_acc)
    pp_desired_torques = pp_inverse_dynamics(dp_one_params, dp_two_params, pp_angular)
    
    returnval[1] = pp_desired_torques.get('one_torque')
    returnval[2] = pp_desired_torques.get('two_torque')
    
    if(returnval[0] > 20.0):
        returnval[0] = 20.0
    if(returnval[0] < -20.0):
        returnval[0] = -20.0
    if(returnval[1] > 20.0):
        returnval[1] = 20.0
    if(returnval[1] < -20.0):
        returnval[1] = -20.0
    if(returnval[2] > 20.0):
        returnval[2] = 20.0
    if(returnval[2] < -20.0):
        returnval[2] = -20.0
#    if t > 0.5 and t < 0.75:
#        returnval[2] = returnval[2] + 0.1
#    print(returnval)
    torque_vector.append(returnval)
    time_vector.append(t)
    return returnval

def test_control(x,t):
    return [1.0, 2.0, 3.0]

def draw_init():
    x1 = -1*numerical_constants[0]*np.sin(x0[0])
    y1 = numerical_constants[0]*np.cos(x0[0])
    x2 = x1 + -1*numerical_constants[2]*np.sin(x0[0] + x0[1])
    y2 = y1 + numerical_constants[2]*np.cos(x0[0] + x0[1])
    x3 = x2 + -1*numerical_constants[4]*np.sin(x0[0] + x0[1] + x0[2])
    y3 = y2 + numerical_constants[4]*np.cos(x0[0] + x0[1] + x0[2])
    com_loc = get_global_com_pos(x0)
    com_bc_loc = get_dp_bc_com_loc(x0)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto', autoscale_on=False,
                     xlim=(-.06, .05), ylim=(-0.5, 2.5))
    plt.plot([0, x1, x2, x3], [0, y1, y2, y3], 'o-', lw=2)
    plt.plot([0, com_loc[0]], [0, com_loc[1]], 'o-', lw=2, c='r')
    plt.plot([x1, x1+com_bc_loc[0]], [y1, y1+com_bc_loc[1]], 'o-', lw=2, c='g')
    plt.show()

args['specified'] = controller

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

x1 = -1*numerical_constants[0]*np.sin(y[:,0])
y1 = numerical_constants[0]*np.cos(y[:,0])

x2 = x1 + -1*numerical_constants[2]*np.sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[2]*np.cos(y[:,0] + y[:,1])

x3 = x2 + -1*numerical_constants[4]*np.sin(y[:,0] + y[:,1] + y[:,2])
y3 = y2 + numerical_constants[4]*np.cos(y[:,0] + y[:,1] + y[:,2])

com_vector = []
com_bc_vector = []

for element in y:
    com_vector.append(get_global_com_pos(element))
    com_bc_vector.append(get_dp_bc_com_loc(element))

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto', autoscale_on=False,
                     xlim=(-0.06, 0.05), ylim=(-0.5, 2.5))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
com_line,  = ax.plot([], [], 'o-', lw=2, c='r')
com_bc_line, = ax.plot([], [], 'o-', lw=2, c = 'g')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
state_text = ax.text(.55, 0.9, '', transform = ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    com_line.set_data([], [])
    com_bc_line.set_data([],[])
    time_text.set_text('')
    state_text.set_text('')
    return line,com_line,com_bc_line,state_text, time_text

def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]
    comx = [0, com_vector[i][0]]
    comy = [0, com_vector[i][1]]
    combcx = [x1[i], x1[i]+com_bc_vector[i][0]]
    combcy = [y1[i], y1[i]+com_bc_vector[i][1]]
    line.set_data(thisx, thisy)
    com_line.set_data(comx, comy)
    com_bc_line.set_data(combcx, combcy)
    time_text.set_text('time = %.1f' % (i*dt))
    state_text.set_text('t1=%.3f, t2=%.3f, t3=%.3f \ns1=%.3f, s2=%.3f, s3=%.3f' %(y[i][0],y[i][1],y[i][2], y[i][3], y[i][4], y[i][5]))
    
    return line, com_line, com_bc_line, state_text, time_text

# choose the interval based on dt and the time to animate one step
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000, blit=True, init_func=init)
#ani.save('triple_pendulum_equivalent_controlled_initial_results.mp4')

plt.show()

