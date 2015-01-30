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
import double_pendulum_particle_setup as pp
import double_pendulum_rb_setup as rb
import triple_pendulum_setup as tp
import pickle



numerical_constants = array([0.5,
                             0.5,
                             0.75,
                             0.75,
                             1.0,
                             1.0,
                             9.81])

parameter_dict = dict(zip(tp.constants, numerical_constants))


gravity_dict = dict(zip([sp.g, pp.g, rb.g, tp.g], [9.81,9.81, 9.81, 9.81]))
global_mass_dict = dict(zip([sp.s_mass], [tp.a_mass+tp.b_mass+tp.c_mass]))
sp_forcing = -1*simplify(sp.forcing - sp.s_torque) + sp.s_torque
sp_eq = (sp.mass_matrix.inv()*sp_forcing)[0].subs(gravity_dict)
sp_eq_torque_only = sp.s_torque*(sp.mass_matrix.inv())[0].subs(global_mass_dict).subs(parameter_dict)
sp_eq = sp_eq.subs(global_mass_dict).subs(parameter_dict)

sp_eq_string = str(sp_eq)
sp_eq_vars = str([sp.s_length, sp.theta_s, sp.s_torque])
file = open('sp_eq.pkl', 'wb')
pickle.dump([sp_eq_string, sp_eq_vars], file)
file.close()

sp_eq_torque_only_string = str(sp_eq)
sp_eq_torque_only_vars = str([sp.s_length, sp.s_torque])
file = open('sp_eq_torque_only.pkl', 'wb')
pickle.dump([sp_eq_torque_only_string, sp_eq_torque_only_vars], file)
file.close()


com_eq_string = str(tp.com.subs(parameter_dict))
com_eq_vars = str((tp.theta_a, tp.theta_b, tp.theta_c))
file = open('com_eq.pkl', 'wb')
pickle.dump([com_eq_string, com_eq_vars], file)
file.close()



com_dot_eq_string = str(tp.com_dot.subs(parameter_dict))
com_dot_eq_vars = str([tp.theta_a, tp.theta_b, tp.theta_c, 
                           tp.omega_a, tp.omega_b, tp.omega_c])
file = open('com_dot_eq.pkl', 'wb')
pickle.dump([com_dot_eq_string, com_dot_eq_vars], file)
file.close()


solved_pp_com_ddot_t1_string = str(pp.des_t1_acc)
solved_pp_com_ddot_t1_vars = str([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y])
file = open('solved_pp_com_ddot_t1.pkl', 'wb')
pickle.dump([solved_pp_com_ddot_t1_string, solved_pp_com_ddot_t1_vars], file)
file.close()


solved_pp_com_ddot_t2_string = str(pp.des_t2_acc)
solved_pp_com_ddot_t2_vars = str([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.des_com_ang_acc_x, pp.des_com_ang_acc_y])
file = open('solved_pp_com_ddot_t2.pkl', 'wb')
pickle.dump([solved_pp_com_ddot_t2_string, solved_pp_com_ddot_t2_vars], file)
file.close()


solved_rb_com_ddot_t1_string = str(rb.des_t1_acc)
solved_rb_com_ddot_t1_vars = str([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.des_com_ang_acc_x, rb.des_com_ang_acc_y])
file = open('solved_rb_com_ddot_t1.pkl', 'wb')
pickle.dump([solved_rb_com_ddot_t1_string, solved_rb_com_ddot_t1_vars], file)
file.close()


solved_rb_com_ddot_t2_string = str(rb.des_t2_acc)
solved_rb_com_ddot_t2_vars = str([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.des_com_ang_acc_x, rb.des_com_ang_acc_y])
file = open('solved_rb_com_ddot_t2.pkl', 'wb')
pickle.dump([solved_rb_com_ddot_t2_string, solved_rb_com_ddot_t2_vars], file)
file.close()

bc_com_string = str(tp.com_bc_in_a.subs(parameter_dict))
bc_com_vars = str([tp.theta_b, tp.theta_c])
file = open('bc_com.pkl', 'wb')
pickle.dump([bc_com_string, bc_com_vars], file)
file.close()


bc_com_inertial_string = str(tp.com_bc.subs(parameter_dict))
bc_com_inertial_vars = str([tp.theta_a, tp.theta_b, tp.theta_c])
file = open('bc_com_inertial.pkl', 'wb')
pickle.dump([bc_com_inertial_string, bc_com_inertial_vars], file)
file.close()


bc_com_dot_string = str(tp.com_bc_dot_in_a.subs(parameter_dict))
bc_com_dot_vars = str([tp.theta_b, tp.theta_c, tp.omega_b, tp.omega_c])
file = open('bc_com_dot.pkl', 'wb')
pickle.dump([bc_com_dot_string, bc_com_dot_vars], file)
file.close()

a_com_string = str(tp.com_a.subs(parameter_dict))
a_com_vars = str([tp.theta_a])
file = open('a_com.pkl', 'wb')
pickle.dump([a_com_string, a_com_vars], file)
file.close()

a_com_dot_string = str(tp.com_a_dot.subs(parameter_dict))
a_com_dot_vars = str([tp.theta_a, tp.omega_a])
file = open('a_com_dot.pkl', 'wb')
pickle.dump([a_com_dot_string, a_com_dot_vars], file)
file.close()

pp_inverse_dynamics_string = str( pp.inverse_dynamics.subs(gravity_dict))
pp_inverse_dynamics_vars = str([pp.one_length, pp.two_length, pp.one_mass, pp.two_mass, pp.theta1, pp.theta2, pp.omega1, pp.omega2, pp.alpha1, pp.alpha2])
file = open('pp_inverse_dynamics.pkl', 'wb')
pickle.dump([pp_inverse_dynamics_string, pp_inverse_dynamics_vars], file)
file.close()


rb_inverse_dynamics_string = str(rb.inverse_dynamics.subs(gravity_dict))
rb_inverse_dynamics_vars = str([rb.one_length, rb.one_com_x, rb.one_com_y, rb.two_com_x, rb.two_com_y, rb.one_com_x_dot, rb.one_com_y_dot, rb.two_com_x_dot, rb.two_com_y_dot, rb.one_mass, rb.two_mass, rb.theta1, rb.theta2, rb.omega1, rb.omega2, rb.alpha1, rb.alpha2])
file = open('rb_inverse_dynamics.pkl', 'wb')
pickle.dump([rb_inverse_dynamics_string, rb_inverse_dynamics_vars], file)
file.close()

args_ode_funcs = str([tp.mass_matrix_full, tp.forcing_vector_full, 
                                        tp.constants, tp.coordinates,
                                        tp.speeds, tp.specified])
file = open('args_ode_funcs.pkl', 'wb')
pickle.dump(args_ode_funcs, file)
file.close()


