from numpy import array, zeros
from sympy import symbols, simplify, trigsimp, cos, sin, Matrix, solve, sympify
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, kinetic_energy, potential_energy
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()

def dq(r, d):
    return [Matrix(r), Matrix(d)]

def dq_point(x, y, z):
    return [Matrix([0, 0, 0, 1]), Matrix([x, y, z, 0])]

def dq_trans(x, y, z):
    return [Matrix([0, 0, 0, 1]), Matrix([x/2, y/2, z/2, 0])]

def dq_rot(x, y, z, theta):
    return [Matrix([x*sin(theta/2), y*sin(theta/2), z*sin(theta/2), cos(theta/2)]), Matrix([0, 0, 0, 0])]

def q_mult(q, p):
    q_big = Matrix([[ q[3], -q[2],  q[1],  q[0]],
                    [ q[2],  q[3], -q[0],  q[1]],
                    [-q[1],  q[0],  q[3],  q[2]],
                    [-q[0], -q[1], -q[2],  q[3]]])
    return simplify(q_big*p)

def dq_mult(q, p):
    result_r = simplify(Matrix([q_mult(q[0], p[0])]))
    qr_plus_pd = q_mult(q[0], p[1])
    qd_plus_pr = q_mult(q[1], p[0])
    result_d = qr_plus_pd + qd_plus_pr
    return [result_r, result_d]

def q_conj(q):
    return simplify(Matrix([-q[0], -q[1], -q[2], q[3]]))

def dq_conj(q):
    return [simplify(q_conj(q[0])), simplify(-q_conj(q[1]))]
 
def dq_extr_trans(q):
    d = q[0]
    r_star = q_conj(q[1])
    dcr = q_mult(d, r_star)
    return simplify(2*dcr)

def dq_add(q, p):
    return [q[0]+p[0], q[1]+p[1]]

def dq_smult(q, s):
    return [s*q[0], s*q[1]]
    
def dq_diff(q, t):
    return [q[0].diff(t), q[1].diff(t)]

def dq_normalize(q):
    r = q[0]
    d = q[1]
    norm = (r[0]**2 + r[1]**2 + r[2]**2 + r[3]**2)**0.5
    r = r/norm
    d = d/norm
    return [r, d]
