from double_pendulum_particle_virtual_setup import *
from sympy import expand, atan2
x, y, xdot, ydot, xddot, yddot = dynamicsymbols('x, y, xdot, ydot, xddot, yddot')

xydot_dict = dict(zip([x.diff(time), y.diff(time)], [xdot, ydot]))
xyddot_dict = dict(zip([xdot.diff(time), ydot.diff(time)], [xddot, yddot]))
xycom_dict = dict(zip([x, y], [com[0], com[1]]))
xycom_dot_dict = dict(zip([xdot, ydot], [com_dot[0], com_dot[1]]))

t_c = -atan2(x, y)
t_c_dot = t_c.diff(time).subs(xydot_dict)
t_c_ddot = t_c_dot.diff(time).subs(xydot_dict).subs(xyddot_dict)

t_c_xddot = t_c_ddot.diff(xddot)
t_c_yddot = t_c_ddot.diff(yddot)
t_c_oddot = simplify(expand(t_c_ddot) - expand(xddot*t_c_xddot) - expand(yddot*t_c_yddot))

t_c_ddot_mat = simplify(Matrix([[t_c_xddot, t_c_yddot, t_c_oddot]]).subs(xycom_dict).subs(xycom_dot_dict))

acc = Matrix(accelerations)
zero_t1 = dict(zip([one_torque], [0]))
eom = simplify(kane.mass_matrix*acc - kane.forcing + torques)

eom_with_torques = simplify(kane.mass_matrix*acc - kane.forcing.subs(zero_t1))

alpha1_virt = solve(eom_with_torques[0], [alpha1])[0]

alpha2_virt = solve(eom_with_torques[1], [alpha2])[0]

alpha1_virt_dict = dict(zip([alpha1], [alpha1_virt]))

alpha1_zero_dict = dict(zip([alpha1], [0]))
alpha2_virt = simplify(alpha2_virt.subs(alpha1_zero_dict))

alpha1_virt_a2 = alpha1_virt.diff(alpha2)
alpha1_virt_t2 = alpha1_virt.diff(two_torque)
alpha2_virt_a2 = alpha2_virt.diff(alpha2)
alpha2_virt_t2 = alpha2_virt.diff(two_torque)

alpha1_virt_other = simplify(expand(alpha1_virt) - expand(alpha2*alpha1_virt_a2) - expand(two_torque*alpha1_virt_t2))
alpha2_virt_other = simplify(expand(alpha2_virt) - expand(alpha2*alpha2_virt_a2) - expand(two_torque*alpha2_virt_t2))

alpha_virt_mat = Matrix([[alpha1_virt_a2, alpha1_virt_t2, alpha1_virt_other], [alpha2_virt_a2, alpha2_virt_t2, alpha2_virt_other]])
alpha_virt_mat_padded = alpha_virt_mat.col_join(Matrix([[0, 0, 1]]))

cdd_xa1 = com_ddot[0].diff(alpha1)
cdd_xa2 = com_ddot[0].diff(alpha2)
cdd_xo = simplify(expand(com_ddot[0]) - expand(alpha1*cdd_xa1) - expand(alpha2*cdd_xa2))
cdd_ya1 = com_ddot[1].diff(alpha1)
cdd_ya2 = com_ddot[1].diff(alpha2)
cdd_yo = simplify(expand(com_ddot[1]) - expand(alpha1*cdd_ya1) - expand(alpha2*cdd_ya2))

com_ddot_mat = Matrix([[cdd_xa1, cdd_xa2, cdd_xo], [cdd_ya1, cdd_ya2, cdd_yo]])

alpha_cdd_mat = simplify(com_ddot_mat*alpha_virt_mat_padded)
alpha_cdd_mat_padded = alpha_cdd_mat.col_join(Matrix([[0, 0, 1]]))

tcdd_alpha_cdd_mat = simplify(t_c_ddot_mat*alpha_cdd_mat_padded)

alpha2_reverse_mat = Matrix([[1/tcdd_alpha_cdd_mat[0], -tcdd_alpha_cdd_mat[1]/tcdd_alpha_cdd_mat[0], -tcdd_alpha_cdd_mat[1]/tcdd_alpha_cdd_mat[0]]])
alpha2_reverse_mat_coefs = Matrix([des_com_ang_acc, two_torque, 1])
alpha2_reverse_mat_padded = alpha2_reverse_mat.col_join(Matrix([[0, 0, 1]]))

zero_force = dict(zip([F_com_x, F_com_y], [0, 0]))
inv_dyn = eom.subs(zero_force)

inv_dyn_2 = inv_dyn[1].subs(alpha1_virt_dict)
inv_dyn_2_a2 = inv_dyn_2.diff(alpha2)
inv_dyn_2_other = simplify(expand(inv_dyn_2) - expand(alpha2*inv_dyn_2_a2))

inv_dyn_2_mat = Matrix([[inv_dyn_2_a2, inv_dyn_2_other]])

t2_mat = simplify(inv_dyn_2_mat*alpha2_reverse_mat_padded)

t2 = (t2_mat[0]*des_com_ang_acc + t2_mat[2])/(1-t2_mat[1])

