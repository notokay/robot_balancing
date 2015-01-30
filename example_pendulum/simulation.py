from pydy.viz.shapes import Cylinder, Sphere
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene


from pendulum import *

# Specify Numerical Quantities
# ============================

initial_coordinates = [deg2rad(0.0), deg2rad(-160)]

initial_speeds = deg2rad(0) * ones(len(speeds))
x0 = concatenate((initial_coordinates, initial_speeds), axis=1)

# taken from male1.txt in yeadon (maybe I should use the values in Winters).
numerical_constants = array([1.0,  # lower_leg_length [m]
                             0.5,  # lower_leg_com_length [m]
                             1.0,  # lower_leg_mass [kg]
                             1.0,  # lower_leg_inertia [kg*m^2]
                             1.0,  # upper_leg_length [m]
                             0.5,  # upper_leg_com_length
                             1.0,  # upper_leg_mass [kg]
                             1.0,  # upper_leg_inertia [kg*m^2]
                             9.81],  # acceleration due to gravity [m/s^2]
                           )

args = {'constants': numerical_constants,
        'specified': array([0.0, 0.0])}

# Simulate
# ========

frames_per_sec = 60
final_time = 5.0

t = linspace(0.0, final_time, final_time * frames_per_sec)

y = odeint(right_hand_side, x0, t, args=(args,))

ankle_shape = Sphere(color='black', radius=0.1)
knee_shape = Sphere(color='black', radius=0.1)
hip_shape = Sphere(color='black', radius=0.1)

ankle_viz_frame = VisualizationFrame(inertial_frame, ankle, ankle_shape)
knee_viz_frame = VisualizationFrame(inertial_frame, knee, knee_shape)

hip = Point('H')
hip.set_pos(knee, upper_leg_length*upper_leg_frame.y)
hip_viz_frame = VisualizationFrame(inertial_frame, hip, hip_shape)

constants_dict = dict(zip(constants, numerical_constants))

lower_leg_center = Point('l_c')
upper_leg_center = Point('u_c')

lower_leg_center.set_pos(ankle, lower_leg_length / 2 * lower_leg_frame.y)
upper_leg_center.set_pos(knee, upper_leg_length / 2 * upper_leg_frame.y)

lower_leg_shape = Cylinder(radius=0.08,
                           length=constants_dict[lower_leg_length],
                           color='blue')
lower_leg_viz_frame = VisualizationFrame('Lower Leg', lower_leg_frame,
                                         lower_leg_center, lower_leg_shape)

upper_leg_shape = Cylinder(radius=0.08,
                           length=constants_dict[upper_leg_length],
                           color='green')
upper_leg_viz_frame = VisualizationFrame('Upper Leg', upper_leg_frame,
                                         upper_leg_center, upper_leg_shape)


scene = Scene(inertial_frame, ankle,
              ankle_viz_frame, knee_viz_frame, hip_viz_frame,
              lower_leg_viz_frame, upper_leg_viz_frame)

scene.generate_visualization_json(coordinates + speeds, constants, y,
                                  numerical_constants)

scene.display()
