from sympy import symbols, simplify, trigsimp, solve, latex, diff, cos, sin
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
#from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, rad2deg, ones, concatenate, pi, zeros, dot, eye
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import numpy as np
from sympy.utilities import lambdify
from sympy.physics.vector import init_vprinting, vlatex
import pickle
from body_model_setup import theta1, theta2, theta3, theta4, omega1, omega2, omega3,omega4, l_ankle_torque, l_hip_torque,waist_torque, r_hip_torque, coordinates, speeds, kane, mass_matrix, forcing_vector, specified, parameter_dict, constants, numerical_constants

init_vprinting()

#Create dictionaries for the values for the values
zero_speed_dict = dict(zip(speeds, zeros(len(speeds))))
torque_dict = dict(zip([l_ankle_torque], [0]))

forcing_matrix = kane.forcing

forcing_matrix = simplify(forcing_matrix)

forcing_matrix = simplify(forcing_matrix.subs(zero_speed_dict).subs(parameter_dict).subs(torque_dict))

forcing_solved = solve(forcing_matrix, [l_hip_torque, r_hip_torque,waist_torque, sin(theta1)])

lam_l = lambdify((theta1, theta2, theta3, theta4), forcing_solved[l_hip_torque])

lam_w = lambdify((theta1, theta2, theta3, theta4), forcing_solved[waist_torque])

lam_r = lambdify((theta1, theta2, theta3, theta4), forcing_solved[r_hip_torque])

lam_f = lambdify((theta1, theta2, theta3,theta4), forcing_matrix[0])

print("finished setting up equations, about to start looping")

t1 = -1.57
t2 = -1.57
t3 = -1.57
t4 = -1.57
T1 = []
T2 = []
T3 = []
T4 = []

answer_vector = []
trim = []
threshold = 0.001

while t1 < 0.5:
    t2 = -1.57
    t3 = -1.57
    t4 = -1.57
    while t2 < 1.57:
        t3 = -1.57
        t4 = -1.57
        while t3 < 1.57:
            t4 = -1.57
            while t4 < 1.57:
                lam_sol = lam_f(t1,t2,t3,t4)
                if(lam_sol < threshold and lam_sol > -1*threshold):
                    answer_vector.append([lam_sol,lam_l(t1,t2,t3,t4), lam_r(t1,t2,t3,t4), t1, t2, t3, t4])
                    T1.append(t1)
                    T2.append(t2)
                    T3.append(t3)
                    T4.append(t4)
                    trim.append([lam_l(t1,t2,t3,t4),lam_w(t1,t2,t3,t4), lam_r(t1,t2,t3,t4)])
                t4 = t4 + 0.01
            t3 = t3 + 0.01
        t2 = t2 + 0.01
    print(t1)
    t1 = t1 + 0.01
                    


outputt1 = open('body_model_angle_one.pkl', 'wb')
outputt2 = open('body_model_angle_two.pkl', 'wb')
outputt3 = open('body_model_angle_three.pkl', 'wb')
outputt4 = open('body_model_angle_four.pkl', 'wb')
outputtor = open('body_model_trim.pkl', 'wb')

pickle.dump(T1, outputt1)
pickle.dump(T2, outputt2)
pickle.dump(T3, outputt3)
pickle.dump(T4, outputt4)
pickle.dump(trim, outputtor)

outputt1.close()
outputt2.close()
outputt3.close()
outputt4.close()
outputtor.close()

