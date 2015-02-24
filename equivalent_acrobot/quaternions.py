from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix, solve, sympify, expand
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()

i, j, k = dynamicsymbols('i, j, k')
e = symbols('epsilon')

theta = dynamicsymbols('theta')

l = symbols('l')

simp_dict = dict(zip([i**2, j**2, k**2, i*j, j*k, k*i, j*i, k*j, i*k], [-1, -1, -1,k, i, j, -k, -i, -j]))

q_rot = k*sin(theta/2) + cos(theta/2)
q_trans = e*l*(j/2) + 1

q_sp = expand(q_rot*q_trans).subs(simp_dict)

time = symbols('t')
