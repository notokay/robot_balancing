from double_pendulum_setup import *
import numpy as np
from sympy import Matrix
from sympy.physics.mechanics import dynamicsymbols

forcing_matrix = kane.forcing
mass_matrix = kane.mass_matrix
forcing_matrix[0] = -1*(forcing_matrix[0] - ankle_torque)
forcing_matrix[1] = -1*(forcing_matrix[1] - waist_torque)

b = Matrix(np.flipud(forcing_matrix))
mass_matrix = Matrix(np.flipud(mass_matrix))

qddot_a, qddot_w = dynamicsymbols('qddot_a, qddot_w')

mass_inv = mass_matrix.inv()
W_pp = mass_matrix[3] - mass_matrix[2]*mass_inv[0]*mass_matrix[1]
W_pp = simplify(1/W_pp)

W_pa = -W_pp*mass_matrix[2]*mass_inv[0]

desired_u = dynamicsymbols('u')

tau_a = (mass_matrix[1] - mass_matrix[0]*(1/mass_matrix[2])*mass_matrix[3])*desired_u - mass_matrix[0]*(1/mass_matrix[2])*b[1] + b[0]
