from pydy.viz.shapes import Cylinder, Sphere
import pydy.viz
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
from numpy import array, zeros, concatenate, linspace, dot
from triple_pendulum_setup_alt import inertial_frame, l_ankle, l_hip, r_hip, l_leg_frame, body_frame, r_leg_frame, r_leg_mass_center, l_leg_mass_center, body_mass_center, constants, numerical_constants, l_leg_length, hip_width, coordinates, speeds, mass_matrix, forcing_vector, specified
import pickle
import numpy as np
right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
initial_coordinates = array([0.25,-0.075,-0.17])
initial_speeds = zeros(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)
frames_per_sec = 60
final_time = 5.0
args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}


t = linspace(0.0, final_time, final_time * frames_per_sec)


inputK = open('triple_pen_LQR_K_useful.pkl','rb')
inputa1 = open('triple_pen_angle_one_useful.pkl','rb')
inputa2 = open('triple_pen_angle_two_useful.pkl','rb')
inputa3 = open('triple_pen_angle_three_useful.pkl','rb')
inputt = open('triple_pendulum_trim_zoom_half.pkl','rb')

K = pickle.load(inputK)
angle_1 = pickle.load(inputa1)
a1 = angle_1
angle_1 = np.asarray(angle_1, dtype = float)
angle_2 = pickle.load(inputa2)
a2 = angle_2
angle_2 = np.asarray(angle_2, dtype = float)
angle_3 = pickle.load(inputa3)
a3 = angle_3
angle_3 = np.asarray(angle_3, dtype =float)
trim = pickle.load(inputt)
trim = np.asarray(trim, dtype = float)

inputK.close()
inputa1.close()
inputa2.close()
inputa3.close()
inputt.close()

torque_vector = []
lastk = []
idx_vector = []
lastidx = 0
counter = 0
tracking_vector = []
curr_vector = []
time_vector = []
output_vector = []
diff_vector = []

def zero_stay_controller(x,t):
  global lastidx
  global counter
  torquelim = 400
  if(counter ==0):
    lastidx = np.abs(angle_1-x[0]).argmin()
    lastidx = lastidx + np.abs(angle_3[lastidx:lastidx +30] - x[2]).argmin()
    counter = counter + 1
    idx = lastidx
    print('first round')
    print(lastidx)
  if(x[3] < 0.5 and x[3] > -0.5):
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_3[idx:idx+30] - x[2]).argmin()
    idx_vector.append(lastidx)
    lastidx = idx
    print('adapt')
    print(idx)
  elif(x[4] > 1.0 and x[4] < -1.0):
    idx = lastidx
    idx_vector.append(lastidx)
    print('stay')
    print(idx)
  else:
    idx = np.abs(angle_1 - x[0]).argmin()
    idx = idx + np.abs(angle_3[idx:idx+30] - x[2]).argmin()
    lastidx = idx
    print(idx)
    idx_vector.append(lastidx)
  returnval = -dot(K[idx],x)
  tracking_vector.append([angle_1[idx], angle_2[idx], angle_3[idx]])
  curr_vector.append([x[0], x[1], x[2]])
  if(returnval[1] > torquelim):
    returnval[1] = torquelim
  if(returnval[1] < -1*torquelim):
    returnval[1] = -1*torquelim
  if(returnval[2] > torquelim):
    returnval[2] = torquelim
  if(returnval[2] < -1*torquelim):
    returnval[2] = -1*torquelim
  returnval[0] = 0
  if(x[0] < -0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  if(x[0] > 0.48):
    returnval[1] = 0
    returnval[2] = 0
    returnval[0] = 0
  torque_vector.append(returnval)
  time_vector.append(t)
  return returnval

args['specified'] = zero_stay_controller


y = odeint(right_hand_side, x0, t, args=(args,))


l_ankle_shape = Sphere(color='black', radius = 0.1)
l_hip_shape = Sphere(color='black', radius = 0.1)
r_hip_shape = Sphere(color='black', radius = 0.1)
r_ankle_shape = Sphere(color='black', radius = 0.1)

l_ankle_viz_frame = VisualizationFrame(inertial_frame, l_ankle, l_ankle_shape)
l_hip_viz_frame = VisualizationFrame(inertial_frame, l_hip, l_hip_shape)
r_hip_viz_frame = VisualizationFrame(inertial_frame, r_hip, r_hip_shape)

r_ankle_viz_frame = VisualizationFrame(inertial_frame, r_leg_mass_center, r_ankle_shape)

constants_dict = dict(zip(constants, numerical_constants))

l_leg_shape = Cylinder(radius = 0.08, length = constants_dict[l_leg_length], color = 'blue')

l_leg_viz_frame = VisualizationFrame('Left Leg', l_leg_frame, l_leg_mass_center, l_leg_shape)

body_shape = Cylinder(radius = 0.08, length = constants_dict[hip_width], color = 'blue')

body_viz_frame = VisualizationFrame('Body', body_frame, body_mass_center, body_shape)

r_leg_shape = Cylinder(radius = 0.08, length = constants_dict[l_leg_length], color = 'red')

r_leg_viz_frame = VisualizationFrame('Right Leg', r_leg_frame, r_leg_mass_center, r_leg_shape)

scene = Scene(inertial_frame, l_ankle)

scene.visualization_frames = [l_ankle_viz_frame,l_hip_viz_frame, r_hip_viz_frame, r_ankle_viz_frame, l_leg_viz_frame, body_viz_frame, r_leg_viz_frame]

scene.generate_visualization_json(coordinates + speeds, constants, y, numerical_constants)

scene.display()
