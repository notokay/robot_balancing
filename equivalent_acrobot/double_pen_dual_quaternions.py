from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix, solve, sympify
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from pydy.codegen.code import generate_ode_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()

time = symbols('t')

theta_d, d_length, omega_d, alpha_d, l_dot, l_ddot = dynamicsymbols('theta_d, l_d, omega_d, alpha_d, l_ddot, l_dddot')

d_i = d_length/2 * sin(theta_d/2)
d_j = d_length/2 * cos(theta_d/2)

r_w = cos(theta_d/2)
r_k = sin(theta_d/2)

thetadot_omega_dict = dict(zip([theta_d.diff(time)], [omega_d]))
ldot_ldot_dict = dict(zip([d_length.diff(time)], [l_dot]))

d_i_dot = d_i.diff(time).subs(thetadot_omega_dict).subs(ldot_ldot_dict)
d_j_dot = d_j.diff(time).subs(thetadot_omega_dict).subs(ldot_ldot_dict)

r_w_dot = r_w.diff(time).subs(thetadot_omega_dict)
r_k_dot = r_k.diff(time).subs(thetadot_omega_dict)

omegadot_alpha_dict = dict(zip([omega_d.diff(time)], [alpha_d]))
lddot_lddot_dict = dict(zip([l_dot.diff(time)], [l_ddot]))

d_i_ddot = d_i_dot.diff(time).subs(thetadot_omega_dict).subs(omegadot_alpha_dict).subs(ldot_ldot_dict).subs(lddot_lddot_dict)
d_j_ddot = d_j_dot.diff(time).subs(thetadot_omega_dict).subs(omegadot_alpha_dict).subs(ldot_ldot_dict).subs(lddot_lddot_dict)

r_w_ddot = r_w_dot.diff(time).subs(thetadot_omega_dict).subs(omegadot_alpha_dict)
r_k_ddot = r_k_dot.diff(time).subs(thetadot_omega_dict).subs(omegadot_alpha_dict)

s_di, s_dj, s_rw, s_rk = dynamicsymbols('s_di, s_dj, s_rw, s_rk')

q = Matrix([d_i_ddot - s_di, d_j_ddot - s_dj, r_w_ddot - s_rw, r_k_ddot - s_rk])

omega_d_solved = solve(q[2], [omega_d])[0]

omega_d_solved_dict = dict(zip([omega_d], [omega_d_solved]))

alpha_d_solved = solve(q[3].subs(omega_d_solved_dict), [alpha_d])[0]

l_dot_solved = solve(q[0], [l_dot])

l_dot_solved_dict = dict(zip([l_dot], l_dot_solved))

l_ddot_solved = solve(q[1].subs(l_dot_solved_dict), [l_ddot])[0]

