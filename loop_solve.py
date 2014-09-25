from sympy import symbols, simplify, trigsimp, solve, latex, cos, sin
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate,pi, zeros, dot, eye
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()
from mpl_toolkits.mplot3d.axes3d import Axes3D

x = -1.57
y = -1.57
z = -1.57
X = []
Y = []
Z = []

answer_vector = []
angle_vector = []

while x < 1.58:
    z = 0
    y = 0
    while y < 1.58:
        z = 0
        while z < 1.58:
            answer_vector.append(lam_f(x,y,z))
            angle_vector.append([x, y, z])
            Z.append(z)
            X.append(x)
            Y.append(y)
            z = z + 0.001
        y = y + 0.001
    x = x + 0.001


