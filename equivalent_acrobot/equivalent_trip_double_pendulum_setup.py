import double_pendulum_rb_setup as rb
import double_pendulum_particle_setup as pp
import triple_pendulum_setup as tr
from sympy import simplify, trigsimp, expand

coordinates_equiv_dict_bc = dict(zip(pp.coordinates, [tr.theta_a + tr.theta_b, tr.theta_c]))

constants_equiv_dict_bc = dict(zip(pp.constants, [tr.b_length, tr.b_mass, tr.c_length, tr.c_mass, tr.g])) 

g_bc = pp.g_terms.subs(coordinates_equiv_dict_bc).subs(constants_equiv_dict_bc)
g_bc = g_bc.simplify()

#constants = [one_length,
#             one_com_x,
#             one_com_y,
#             one_mass,
#             two_com_x,
#             two_com_y,
#             two_mass,
#             g]

coordinates_equiv_dict_a = dict(zip(rb.coordinates, [tr.theta_a, tr.theta_b+tr.theta_c]))

constants_equiv_dict_a = dict(zip(rb.constants, [tr.a_length, tr.com_a[0], tr.com_a[1], tr.a_mass, tr.com_bc[0], tr.com_bc[1], tr.b_mass+tr.c_mass, tr.g]))

g_a = rb.g_terms.subs(coordinates_equiv_dict_a).subs(constants_equiv_dict_a)
g_a = simplify(g_a)


