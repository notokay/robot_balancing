# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import array, zeros, eye, asarray, dot, rad2deg, deg2rad, linspace, sin, cos, pi, concatenate, dot
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy import symbols, simplify, trigsimp, solve, asin, acos, lambdify
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting, vlatex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
from utils import det_controllable
import pickle
#from utils import controllable

init_vprinting()

inputA = open('double_block_linearized_A_full.pkl', 'rb')
inputB = open('double_block_linearized_B_full.pkl', 'rb')
inputa2 = open('double_block_angle_two.pkl','rb')

linearA = pickle.load(inputA)
linearB = pickle.load(inputB)
angle_2 = pickle.load(inputa2)
inputA.close()
inputB.close()
inputa2.close()

B = []

for b in linearB:
    B.append(b[:,1].reshape(4,1))
    

controllability_det = []

for a,b in zip(linearA,B):
    controllability_det.append(det_controllable(a,b))

fig = plt.figure()
plt.grid(True)
plt.scatter(angle_2, controllability_det)
plt.show()
