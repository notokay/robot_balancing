import double_pendulum_rb_setup as rb
import double_pendulum_particle_setup as pp
import quad_pendulum_setup as qd
from sympy import simplify, trigsimp, expand

coordinates_equiv_dict = dict(zip(pp.coordinates, [qd.theta_a + qd.theta_b + qd.theta_c, qd.theta_d]))

constants_equiv_dict = dict(zip(pp.constants, [qd.c_length, qd.c_mass, qd.d_length, qd.d_mass, qd.g])) 

g_dc = pp.g_terms.subs(coordinates_equiv_dict).subs(constants_equiv_dict)
g_dc = g_dc.simplify()

#constants = [one_length,
#             one_com_x,
#             one_com_y,
#             one_mass,
#             two_com_x,
#             two_com_y,
#             two_mass,
#             g]

coordinates_equiv_dict = dict(zip(rb.coordinates, [qd.theta_a+qd.theta_b, qd.theta_c + qd.theta_d]))

constants_equiv_dict = dict(zip(rb.constants, [qd.b_length, qd.com_ba[0], qd.com_ba[1], qd.b_mass, qd.com_cd[0], qd.com_cd[1], qd.c_mass + qd.d_mass, qd.g]))

g_b = rb.g_terms.subs(coordinates_equiv_dict).subs(constants_equiv_dict)
g_b = simplify(g_b)

coordinates_equiv_dict = dict(zip(rb.coordinates, [qd.theta_a, qd.theta_b+qd.theta_c+qd.theta_d]))

constants_equiv_dict = dict(zip(rb.constants, [qd.a_length, qd.com_a[0], qd.com_a[1], qd.a_mass, qd.com_bcd[0], qd.com_bcd[1], qd.b_mass+qd.c_mass+qd.d_mass, qd.g]))

g_c = rb.g_terms.subs(coordinates_equiv_dict).subs(constants_equiv_dict)
g_c = simplify(g_c)


