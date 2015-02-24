from numpy import sin, cos, cumsum, dot, zeros
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.physics.mechanics import kinetic_energy
from numpy import array, linspace, deg2rad, ones, concatenate, sign
from sympy import lambdify, atan, atan2, Matrix, simplify, sympify, tan
from sympy.mpmath import norm
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from double_pendulum_particle_setup import *

d_x, d_y, r_w, r_z = dynamicsymbols('d_x, d_y, r_w, r_z')

time = symbols('t')

zero_params = dict(zip([d_x, d_y, r_w, r_z], [0.0,0.0,0.0,0.0]))

zx_a1 = sympify('((l_1*m_1*cos(theta1(t)/2 - theta2(t)/2) + l_1*m_2*cos(theta1(t)/2 - theta2(t)/2) - l_2*m_2*cos(theta1(t)/2 + theta2(t)/2))*(omega1(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 2*omega1(t)*omega2(t)*sin(theta1(t)/2 + theta2(t)/2) + omega2(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 4*r_z(t)) + (l_1*m_1*omega1(t)**2*sin(theta1(t)/2 - theta2(t)/2) - 2*l_1*m_1*omega1(t)*omega2(t)*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_1*omega2(t)**2*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_2*omega1(t)**2*sin(theta1(t)/2 - theta2(t)/2) - 2*l_1*m_2*omega1(t)*omega2(t)*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_2*omega2(t)**2*sin(theta1(t)/2 - theta2(t)/2) + l_2*m_2*omega1(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 2*l_2*m_2*omega1(t)*omega2(t)*sin(theta1(t)/2 + theta2(t)/2) + l_2*m_2*omega2(t)**2*sin(theta1(t)/2 + theta2(t)/2) - 8*m_1*d_x(t) - 8*m_2*d_x(t))*cos(theta1(t)/2 + theta2(t)/2))/(4*l_1*(m_1 + m_2)*cos(theta1(t)/2 - theta2(t)/2)*cos(theta1(t)/2 + theta2(t)/2))')

zx_a2 = sympify('((l_1*m_1*cos(theta1(t)/2 - theta2(t)/2) + l_1*m_2*cos(theta1(t)/2 - theta2(t)/2) + l_2*m_2*cos(theta1(t)/2 + theta2(t)/2))*(omega1(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 2*omega1(t)*omega2(t)*sin(theta1(t)/2 + theta2(t)/2) + omega2(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 4*r_z(t)) - (l_1*m_1*omega1(t)**2*sin(theta1(t)/2 - theta2(t)/2) - 2*l_1*m_1*omega1(t)*omega2(t)*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_1*omega2(t)**2*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_2*omega1(t)**2*sin(theta1(t)/2 - theta2(t)/2) - 2*l_1*m_2*omega1(t)*omega2(t)*sin(theta1(t)/2 - theta2(t)/2) + l_1*m_2*omega2(t)**2*sin(theta1(t)/2 - theta2(t)/2) + l_2*m_2*omega1(t)**2*sin(theta1(t)/2 + theta2(t)/2) + 2*l_2*m_2*omega1(t)*omega2(t)*sin(theta1(t)/2 + theta2(t)/2) + l_2*m_2*omega2(t)**2*sin(theta1(t)/2 + theta2(t)/2) - 8*m_1*d_x(t) - 8*m_2*d_x(t))*cos(theta1(t)/2 + theta2(t)/2))/(4*l_1*(m_1 + m_2)*cos(theta1(t)/2 - theta2(t)/2)*cos(theta1(t)/2 + theta2(t)/2))')

c_func = sympify('(omega1(t)**2*sin(theta1(t))/2 + omega1(t)*omega2(t)*sin(theta2(t)) + omega2(t)**2*sin(theta1(t))/2)/(cos(theta1(t)) + cos(theta2(t)))')

bk_func = sympify('(-omega1(t)**2*sin(theta1(t))/2 + omega1(t)*omega2(t)*sin(theta2(t)) - omega2(t)**2*sin(theta1(t))/2)/(-cos(theta1(t)) + cos(theta2(t)))')
b_func = sympify('(-l_1*m_1*omega1(t)**2*cos(theta1(t)) + 2*l_1*m_1*omega1(t)*omega2(t)*cos(theta2(t)) - l_1*m_1*omega2(t)**2*cos(theta1(t)) - l_1*m_2*omega1(t)**2*cos(theta1(t)) + 2*l_1*m_2*omega1(t)*omega2(t)*cos(theta2(t)) - l_1*m_2*omega2(t)**2*cos(theta1(t)) - l_2*m_2*omega1(t)**2 - 2*l_2*m_2*omega1(t)*omega2(t) - l_2*m_2*omega2(t)**2)/(2*l_1*(m_1 + m_2)*(sin(theta1(t)) - sin(theta2(t))))')

s_func = 1/(cos(theta1/2 + 3*theta2/2) - cos(3*theta1/2 + theta2/2))

lam_a1 = lambdify([theta1, theta2, omega1, omega2],zx_a1.subs(parameter_dict).subs(zero_params))
lam_a2 = lambdify([theta1, theta2, omega1, omega2],zx_a2.subs(parameter_dict).subs(zero_params))
lam_s = lambdify([theta1, theta2], s_func)
lam_c = lambdify([theta1, theta2, omega1, omega2], c_func)
lam_b = lambdify([theta1, theta2, omega1, omega2], b_func.subs(parameter_dict))
lam_blah = lambdify([theta1, theta2], (1/(-1 + sin(theta2)/tan(theta1) + cos(theta1+theta2))))

a1_val = []
ab_val = []
for i in np.arange(-3.0, 3.0, 0.1):
    for j in np.arange(-3.0, 3.0, 0.1):
        print(i, j)
#        a1_val.append([i, j, lam_s(i, j)])
        a1_val.append([i, j, lam_c(i, j, sign(i)*0.1, sign(j)*0.1)])
#        ab_val.append([i, j, lam_b(i, j, sign(i)*0.1, sign(j)*0.1)])
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
a1_val = np.array(np.array(a1_val), dtype = float)
ab_val = np.array(np.array(ab_val), dtype = float)
ax.scatter(a1_val[:,0], a1_val[:,1], a1_val[:,2], c = 'r')
#ax.scatter(ab_val[:,0], ab_val[:,1], ab_val[:,2], c = 'g')

plt.show()
