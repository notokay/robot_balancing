from numpy import sin, cos, cumsum, dot, zeros
from numpy import array, linspace, deg2rad, ones, concatenate
from sympy import lambdify, atan, atan2, Matrix, simplify, sympify
from sympy.mpmath import norm
import double_pendulum_particle_setup as dp
import single_pendulum_setup as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

mass_matrix = dp.mass_matrix
parameter_dict = dp.parameter_dict


def get_global_com_params(state):
    com_loc = Matrix

for i in arange(-1.57, 1.57, 0.05):
    for j in arange(-1.57, 1.57, 0.05):
        x = [i, j, 0.0, 0.0]
        
