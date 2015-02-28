from numpy import sin, cos, cumsum, dot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan2
from scipy.integrate import odeint
import numpy as np
import single_pendulum_extendable_force_setup as sp

from double_pendulum_particle_setup import *

from double_pendulum_particle_virtual_setup import F_com_x, F_com_y

t2 = sympify('-thetaddot_c*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_2**2)*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)/(l_1*l_2**2*m_1*m_2*(m_1 + m_2)*(l_1*m_1 + l_1*m_2 + l_2*m_2*cos(theta2(t)))) + (2*l_1*l_2*m_2*(m_1 + m_2)*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)*(l_1**2*m_1**3*omega1(t) + 3*l_1**2*m_1**2*m_2*omega1(t) + 3*l_1**2*m_1*m_2**2*omega1(t) + l_1**2*m_2**3*omega1(t) + 2*l_1*l_2*m_1**2*m_2*omega1(t)*cos(theta2(t)) + l_1*l_2*m_1**2*m_2*omega2(t)*cos(theta2(t)) + 4*l_1*l_2*m_1*m_2**2*omega1(t)*cos(theta2(t)) + 2*l_1*l_2*m_1*m_2**2*omega2(t)*cos(theta2(t)) + 2*l_1*l_2*m_2**3*omega1(t)*cos(theta2(t)) + l_1*l_2*m_2**3*omega2(t)*cos(theta2(t)) + l_2**2*m_1*m_2**2*omega1(t) + l_2**2*m_1*m_2**2*omega2(t) + l_2**2*m_2**3*omega1(t) + l_2**2*m_2**3*omega2(t))*omega2(t)*sin(theta2(t)) + ((l_1*m_1*sin(theta1(t)) + m_2*(l_1*sin(theta1(t)) + l_2*sin(theta1(t) + theta2(t))))*(-m_2*(-g*m_1*sin(theta1(t) + theta2(t)) - g*m_2*sin(theta1(t) + theta2(t)) + l_1*m_1*omega1(t)**2*sin(theta2(t)) + l_1*m_2*omega1(t)**2*sin(theta2(t)) + F_cx(t)*cos(theta1(t) + theta2(t)) + F_cy(t)*sin(theta1(t) + theta2(t)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)*sin(theta1(t) + theta2(t)) + (m_1 + m_2)*(l_1*m_1*sin(theta1(t)) + m_2*(l_1*sin(theta1(t)) + l_2*sin(theta1(t) + theta2(t))))*(g*l_1*m_1**2*sin(theta1(t)) + 2*g*l_1*m_1*m_2*sin(theta1(t)) + g*l_1*m_2**2*sin(theta1(t)) + g*l_2*m_1*m_2*sin(theta1(t) + theta2(t)) + g*l_2*m_2**2*sin(theta1(t) + theta2(t)) + 2*l_1*l_2*m_1*m_2*omega1(t)*omega2(t)*sin(theta2(t)) + l_1*l_2*m_1*m_2*omega2(t)**2*sin(theta2(t)) + 2*l_1*l_2*m_2**2*omega1(t)*omega2(t)*sin(theta2(t)) + l_1*l_2*m_2**2*omega2(t)**2*sin(theta2(t)) - l_1*m_1*F_cx(t)*cos(theta1(t)) - l_1*m_1*F_cy(t)*sin(theta1(t)) - l_1*m_2*F_cx(t)*cos(theta1(t)) - l_1*m_2*F_cy(t)*sin(theta1(t)) - l_2*m_2*F_cx(t)*cos(theta1(t) + theta2(t)) - l_2*m_2*F_cy(t)*sin(theta1(t) + theta2(t))) + (m_1 + m_2)*(l_1*m_1*omega1(t)**2*cos(theta1(t)) + l_1*m_2*omega1(t)**2*cos(theta1(t)) + l_2*m_2*omega1(t)**2*cos(theta1(t) + theta2(t)) + 2*l_2*m_2*omega1(t)*omega2(t)*cos(theta1(t) + theta2(t)) + l_2*m_2*omega2(t)**2*cos(theta1(t) + theta2(t)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)) - (l_1*m_1*cos(theta1(t)) + m_2*(l_1*cos(theta1(t)) + l_2*cos(theta1(t) + theta2(t))))*(m_2*(-g*m_1*sin(theta1(t) + theta2(t)) - g*m_2*sin(theta1(t) + theta2(t)) + l_1*m_1*omega1(t)**2*sin(theta2(t)) + l_1*m_2*omega1(t)**2*sin(theta2(t)) + F_cx(t)*cos(theta1(t) + theta2(t)) + F_cy(t)*sin(theta1(t) + theta2(t)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)*cos(theta1(t) + theta2(t)) - (m_1 + m_2)*(l_1*m_1*cos(theta1(t)) + m_2*(l_1*cos(theta1(t)) + l_2*cos(theta1(t) + theta2(t))))*(g*l_1*m_1**2*sin(theta1(t)) + 2*g*l_1*m_1*m_2*sin(theta1(t)) + g*l_1*m_2**2*sin(theta1(t)) + g*l_2*m_1*m_2*sin(theta1(t) + theta2(t)) + g*l_2*m_2**2*sin(theta1(t) + theta2(t)) + 2*l_1*l_2*m_1*m_2*omega1(t)*omega2(t)*sin(theta2(t)) + l_1*l_2*m_1*m_2*omega2(t)**2*sin(theta2(t)) + 2*l_1*l_2*m_2**2*omega1(t)*omega2(t)*sin(theta2(t)) + l_1*l_2*m_2**2*omega2(t)**2*sin(theta2(t)) - l_1*m_1*F_cx(t)*cos(theta1(t)) - l_1*m_1*F_cy(t)*sin(theta1(t)) - l_1*m_2*F_cx(t)*cos(theta1(t)) - l_1*m_2*F_cy(t)*sin(theta1(t)) - l_2*m_2*F_cx(t)*cos(theta1(t) + theta2(t)) - l_2*m_2*F_cy(t)*sin(theta1(t) + theta2(t))) + (m_1 + m_2)*(l_1*m_1*omega1(t)**2*sin(theta1(t)) + l_1*m_2*omega1(t)**2*sin(theta1(t)) + l_2*m_2*omega1(t)**2*sin(theta1(t) + theta2(t)) + 2*l_2*m_2*omega1(t)*omega2(t)*sin(theta1(t) + theta2(t)) + l_2*m_2*omega2(t)**2*sin(theta1(t) + theta2(t)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_1*m_2 + l_2**2*m_2**2)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_2**2))/(l_1*l_2**2*m_1*m_2*(m_1 + m_2)**2*(l_1*m_1 + l_1*m_2 + l_2*m_2*cos(theta2(t)))*(l_1**2*m_1**2 + 2*l_1**2*m_1*m_2 + l_1**2*m_2**2 + 2*l_1*l_2*m_1*m_2*cos(theta2(t)) + 2*l_1*l_2*m_2**2*cos(theta2(t)) + l_2**2*m_2**2))')

# Specify Numerical Quantities
# ============================
right_hand_side = generate_ode_function(mass_matrix_full, forcing_vector, 
                                        constants, coordinates,
                                        speeds, specified)
initial_coordinates = [0., 0.0]

initial_speeds = zeros(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([0.75,
                             7.0,
                             0.5,
                             8.0,
                             9.81],  # acceleration due to gravity [m/s^2]
                           )
parameter_dict = dict(zip(constants, numerical_constants))

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 15.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

right_hand_side(x0, 0.0, args)

x, y = dynamicsymbols('x, y')
lam_com_x = lambdify([theta1, theta2], com[0].subs(parameter_dict))
lam_com_y = lambdify([theta1, theta2], com[1].subs(parameter_dict))
com_angle = -atan2(x, y)
com_vel = com_angle.diff(time)
xy_com_dict = dict(zip([x, y], [com[0], com[1]]))
com_angle = com_angle.subs(xy_com_dict).subs(parameter_dict)
lam_com_angle = lambdify([theta1, theta2], com_angle)
com_length = com.norm().subs(parameter_dict)
lam_com_length = lambdify([theta1, theta2], com_length)
xdot, ydot = dynamicsymbols('xdot, ydot')
xy_com_dot_dict = dict(zip([x.diff(time), y.diff(time)], [com_dot[0], com_dot[1]]))
com_vel = com_vel.subs(xy_com_dot_dict).subs(xy_com_dict).subs(parameter_dict)
lam_com_vel = lambdify([theta1, theta2, omega1, omega2], com_vel)

lam_alpha2 = lambdify([theta1, theta2, omega1, omega2, F_com_x, F_com_y, des_com_ang_acc], alpha2_solved.subs(parameter_dict))

mass_dict = dict(zip([sp.s_mass], [one_mass + two_mass]))

lam_sp_eq = lambdify([sp.theta_s, sp.s_length, sp.s_torque, sp.omega_s, sp.s_length_dot], (sp.forcing/sp.mass_matrix[0]).subs(sp.ldot_dict).subs(mass_dict).subs(parameter_dict))

alpha_dict = dict(zip([sp.alpha_s], [(sp.forcing-sp.s_torque)/sp.mass_matrix[0]]))

sp_com_force = sp.ri.inv()*(sp.com_force.subs(mass_dict).subs(parameter_dict))

g_mat = Matrix([0, g, 0])
b = (sp.ri*g_mat).as_mutable()
b[1] = b[1] - (sp.omega_s**2)*sp.s_length
b = sp.s_mass*b
b[1] = b[1] + sp.force_s
#b = sp.s_mass*b
b = simplify(sp.ri.inv()*b).subs(mass_dict).subs(parameter_dict)

#lam_sp_eq_x = lambdify([sp.theta_s, sp.s_length, sp.omega_s, sp.s_length_dot, sp.force_s, sp.s_torque], sp_com_force[0])
#lam_sp_eq_y = lambdify([sp.theta_s, sp.s_length, sp.omega_s, sp.s_length_dot, sp.force_s, sp.s_torque], sp_com_force[1])

lam_sp_eq_x = lambdify([sp.theta_s, sp.s_length, sp.omega_s, sp.force_s], b[0])
lam_sp_eq_y = lambdify([sp.theta_s, sp.s_length, sp.omega_s, sp.force_s], b[1])

lam_sp_l = lambdify([theta1, theta2, omega1, omega2], com_l_dot.subs(parameter_dict))

lam_t2_invdyn = lambdify([theta1, theta2, omega1, omega2, alpha2], t2_inv_dyn_complex.subs(parameter_dict))

lam_t2 = lambdify([theta1, theta2, omega1, omega2, F_com_x, F_com_y, des_com_ang_acc], t2.subs(parameter_dict))

max_l = lam_com_length(0,0)
#com_acc = sp.kane.mass_matrix.inv()*sp.kane.forcing
#sp_constants_dict = dict(zip([sb.a_mass, sb.g], [one_mass + two_mass, 

#com_angle = atan(com[0]/com[1]).subs(parameter_dict)
#com_length = com.norm().subs(parameter_dict)
#lam_com_angle = lambdify([theta1, theta2], com_angle)
#lam_com_length = lambdify([theta1, theta2], com_length)

k_p_t = 1000.0
k_v_t = 1000.0
k_p_l = 350.0
k_v_l = -1000.0
k_dv = .0

#k_p_t = 00.0
#k_v_t = 00.0
#k_p_l = 0*350.0
#k_v_l = 0*-900.0
#k_dv = 0.0

s_vel_vec = []
s_len_vec = []
s_tor_vec = []
com_xy_vec = []
s_des_vec = []
d_des_vec = []
d_tor_vec = []
s_ang_vec = []
d_vel_vec = []
d_pos_vec = []
time_vector = []
def make_sp_test(x):
    s_length = lam_com_length(x[0], x[1])
    print "length is", s_length
    s_angle = lam_com_angle(x[0], x[1])
    print "angle is", s_angle
    s_ang_vec.append(s_angle)
    s_vel = lam_com_vel(x[0], x[1], x[2], x[3])
    print "vel is", s_vel
    s_vel_vec.append(s_vel)
    s_torque = -k_p*s_angle + -k_v*s_vel
    s_tor_vec.append(s_torque)
    print "desired torque is", s_torque
    s_des_alpha = lam_sp_eq(s_angle, s_length, s_torque)
    s_des_vec.append(s_des_alpha)
    print "desired acceleration is", s_des_alpha
    return s_des_alpha

def make_sp(x):
    d_vel_vec.append([x[2], x[3]])
    d_pos_vec.append([x[0], x[1]])
    s_length = lam_com_length(x[0], x[1])
    s_len_vec.append(s_length)
#    print(s_length)
    s_angle = lam_com_angle(x[0], x[1])
#    print(s_angle)
    s_ang_vec.append(s_angle)
    s_vel = lam_com_vel(x[0], x[1], x[2], x[3])
    s_vel_vec.append(s_vel)
    s_l_vel = lam_sp_l(x[0], x[1] ,x[2], x[3])
    s_torque = (-k_p_t*s_angle + -k_v_t*s_vel)
    s_force = (k_p_l*(s_length - max_l) + -k_v_l*s_l_vel)
    s_tor_vec.append(s_torque)
    s_des_alpha = lam_sp_eq(s_angle, s_length, s_torque, s_vel, s_l_vel)
    s_des_vec.append(s_des_alpha)
#    s_des_x = lam_sp_eq_x(s_angle, s_length, s_vel, s_l_vel, s_force, s_torque)
#    s_des_y = lam_sp_eq_y(s_angle, s_length, s_vel, s_l_vel, s_force, s_torque)
    s_des_x = lam_sp_eq_x(s_angle, s_length, s_vel, s_force)
    s_des_y = lam_sp_eq_y(s_angle, s_length, s_vel, s_force)
#    print(s_des_alpha)
    return [s_des_x, s_des_y, s_des_alpha]

def controller(x, t):
    com_xy_vec.append([lam_com_x(x[0], x[1]), lam_com_y(x[0], x[1])])
    time_vector.append(t)
    desired_sp_acc = make_sp(x)
    print(desired_sp_acc)
#    desired_a2 = lam_alpha2(x[0], x[1], x[2], x[3], desired_sp_acc[0], desired_sp_acc[1], desired_sp_acc[2])
    desired_a2 = lam_t2(x[0], x[1], x[2], x[3], desired_sp_acc[0], desired_sp_acc[1], desired_sp_acc[2])
#    print(desired_a2)
    d_des_vec.append(desired_a2)
    desired_torque = lam_t2_invdyn(x[0], x[1], x[2], x[3], desired_a2)
#    desired_torque = lam_t2(x[0], x[1], x[2], x[3], desired_sp_acc[0], desired_sp_acc[1], desired_sp_acc[2])
#    if(desired_torque > 10):
#        desired_torque = 10.0
#    if(desired_torque < -10):
#        desired_torque = -10.0
    desired_torque = desired_torque + -k_dv*x[3]
    if(desired_torque > 150):
        desired_torque = 150
    if(desired_torque < -150):
        desired_torque = -150
    d_tor_vec.append(desired_torque)
    t1 = 0
    if(t > 0.0 and t < 15.0):
        t1 = -.1
    return [t1, desired_torque-1.0]

args['specified'] = controller

y = odeint(right_hand_side, x0, t, args=(args,))

dt = 1./frames_per_sec

x1 = -1*numerical_constants[0]*np.sin(y[:,0])
y1 = numerical_constants[0]*np.cos(y[:,0])

x2 = x1 + -1*numerical_constants[2]*np.sin(y[:,0] + y[:,1])
y2 = y1 + numerical_constants[2]*np.cos(y[:,0] + y[:,1])

com_x = map(lam_com_x, y[:,0], y[:,1])
com_y = map(lam_com_y, y[:,0], y[:,1])

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
com_line,  = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    com_line.set_data([], [])
    time_text.set_text('')
    return com_line, line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    c_x = [0, com_x[i]]
    c_y = [0, com_y[i]]

    line.set_data(thisx, thisy)
    com_line.set_data(c_x, c_y)
    time_text.set_text('time = %.1f' % (i*dt))
    return com_line, line, time_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(y)),
                              interval=dt*1000 - (t1-t0), blit=True, init_func=init)
#ani.save('dp_helicopter_01.mp4', fps = frames_per_sec)

plt.show()

f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)

ax1.plot(time_vector, s_vel_vec)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('s_vel')
ax1.legend(["s_vel"])

ax3.plot(time_vector, s_des_vec)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('s_des')
ax3.legend(["s_des"])

ax4.plot(time_vector, s_ang_vec)
ax4.set_xlabel('Time[s]')
ax4.set_ylabel('s_ang')
ax4.legend(["s_ang"])

ax5.plot(time_vector,s_len_vec)
ax5.set_xlabel('time')
ax5.set_ylabel('s_len')
ax5.legend(["${}$".format(vlatex(s)) for s in speeds])

ax6.plot(time_vector, d_pos_vec)
ax6.set_xlabel('time')
ax6.set_ylabel('angle')
ax6.legend(["${}$".format(vlatex(c)) for c in coordinates])

ax2.plot(time_vector, d_tor_vec)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('d_tor')
#ax2.legend(["${}$".format(vlatex(s)) for s in speeds])

plt.show()

fig = plt.figure()

t1 = [d[0] for d in d_pos_vec]
v1 = [d[0] for d in d_vel_vec]
plt.plot(s_ang_vec, s_vel_vec)

plt.show()

fig = plt.figure()

t2_plot = [d[1] for d in d_pos_vec]
v2 = [d[1] for d in d_vel_vec]

plt.plot(t2_plot, v2)

plt.show()
