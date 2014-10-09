from pydy.viz.shapes import Cylinder, Sphere
import pydy.viz
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene
from pydy.codegen.code import generate_ode_function
from scipy.integrate import odeint
from numpy import array, zeros, concatenate, linspace
from triple_pendulum_setup_alt import inertial_frame, l_ankle, l_hip, r_hip, l_leg_frame, body_frame, r_leg_frame, r_leg_mass_center, l_leg_mass_center, body_mass_center, constants, numerical_constants, l_leg_length, hip_width, coordinates, speeds, mass_matrix, forcing_vector, specified

right_hand_side = generate_ode_function(mass_matrix, forcing_vector,
                                        constants, coordinates, speeds,
                                        specified)
initial_coordinates = array([0.5,0.5,0.5])
initial_speeds = zeros(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)
frames_per_sec = 60
final_time = 5.0
args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0, 0.0])}


t = linspace(0.0, final_time, final_time * frames_per_sec)


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
