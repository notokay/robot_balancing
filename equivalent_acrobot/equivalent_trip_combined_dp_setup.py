from sympy import symbols, Matrix, simplify,trigsimp, sin, cos, atan
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, Particle, KanesMethod, inertia_of_point_mass
from pydy.codegen.code import generate_ode_function
from numpy import array, linspace, deg2rad, ones, concatenate
from scipy.integrate import odeint
#from sympy import *
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting()


# Orientations
# ============

theta_a, theta_b, theta_c, theta_go, theta_bco = dynamicsymbols('theta_a, theta_b, theta_c, theta_go, theta_bco')

inertial_frame = ReferenceFrame('I')

a_frame = ReferenceFrame('A')
a_frame.orient(inertial_frame, 'Axis', (theta_a, inertial_frame.z))

b_frame = ReferenceFrame('B')
b_frame.orient(a_frame, 'Axis', (theta_b, inertial_frame.z))

c_frame = ReferenceFrame('C')
c_frame.orient(b_frame, 'Axis', (theta_c, inertial_frame.z))

go_frame = ReferenceFrame('G_o')

bco_frame = ReferenceFrame('BC_o')


# Point Locations
# ===============

# Joints
# ------

a_length, b_length, c_length = symbols('l_a, l_b, l_c')
a_mass, b_mass, c_mass = symbols('m_a, m_b, m_c')

A = Point('A')
B = Point('B')
C = Point('C')
D = Point('D')
com_global = Point('G_o')
com_bc = Point('BC_o')

B.set_pos(A, a_length * a_frame.y)
C.set_pos(B, b_length * b_frame.y)
D.set_pos(C, c_length * c_frame.y)
t_ai = simplify(B.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))
t_bi = simplify(C.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))
t_ci = simplify(D.pos_from(A).express(inertial_frame).to_matrix(inertial_frame))

t_ba = simplify(C.pos_from(B).express(inertial_frame).to_matrix(inertial_frame))
t_ca = simplify(D.pos_from(B).express(inertial_frame).to_matrix(inertial_frame))

com_global_coords = (t_ai*a_mass + t_bi*b_mass + t_ci*c_mass)/(a_mass+b_mass+c_mass)
com_bc_coords = (t_ba*b_mass + t_ba*c_mass)/(b_mass+c_mass)

theta_go = atan(com_global_coords[0]/com_global_coords[1])
theta_bco = atan(com_bc_coords[0]/com_bc_coords[1])

go_frame.orient(inertial_frame, 'Axis', (theta_go, inertial_frame.z))
bco_frame.orient(b_frame, 'Axis', (theta_bco, inertial_frame.z))

com_global.set_pos(A, com_global_coords.norm()*go_frame.y)
com_bc.set_pos(B, com_bc_coords.norm()*bco_frame.y)

# Define kinematical differential equations
# =========================================
omega_a, omega_b, omega_c, omega_go, omega_bco = dynamicsymbols('omega_a, omega_b, omega_c, omega_go, omega_bco')
time = symbols('t')
#omega_go = theta_go.diff(time)
#omega_bco = theta_bco.diff(time)

kinematical_differential_equations = [omega_a - theta_a.diff(),
                                      omega_b - theta_b.diff(),
                                      omega_c - theta_c.diff(),
                                      omega_go - theta_]

# Angular Velocities
# ==================
a_frame.set_ang_vel(inertial_frame, omega_a*inertial_frame.z)
b_frame.set_ang_vel(a_frame, omega_b*inertial_frame.z)
c_frame.set_ang_vel(b_frame, omega_c*inertial_frame.z)

go_frame.set_ang_vel(inertial_frame, omega_go*inertial_frame.z)
bco_frame.set_ang_vel(b_frame, omega_bco*inertial_frame.z)

# Linear Velocities
# =================
A.set_vel(inertial_frame, 0)
B.v2pt_theory(A, inertial_frame, a_frame)
C.v2pt_theory(B, inertial_frame, b_frame)
D.v2pt_theory(C, inertial_frame, c_frame)
com_global.v2pt_theory(A, inertial_frame, go_frame)
com_bc.v2pt_theory(B, inertial_frame, bco_frame)

# Mass
# ====

a_mass, b_mass, c_mass = symbols('m_a, m_b, m_c')

#Defines the linkages as particles
bP = Particle('bP', B, a_mass)
cP = Particle('cP', C, b_mass)
dP = Particle('dP', D, c_mass)
goP = Particle('goP', com_global, a_mass+b_mass+c_mass)
bcoP = Particle('bcoP', com_bc, b_mass + c_mass)

#Sets up gravity information and assigns gravity to act on mass centers
g = symbols('g')

b_grav_force_vector = -1*a_mass*g*inertial_frame.y
b_grav_force = (B, b_grav_force_vector)
c_grav_force_vector = -1*b_mass*g*inertial_frame.y
c_grav_force = (C, c_grav_force_vector)
d_grav_force_vector = -1*c_mass*g*inertial_frame.y
d_grav_force = (D, d_grav_force_vector)

go_grav_force_vector = -1*(a_mass+b_mass+c_mass)*inertial_frame.y
go_grav_force = (com_global, go_grav_force_vector)
bco_grav_force_vector = -1*(b_mass+c_mass)*inertial_frame.y
bco_grav_force = (com_bc, bco_grav_force_vector)


# Joint Torques
# =============
a_torque, b_torque, c_torque, go_torque, bco_torque = dynamicsymbols('T_a, T_b, T_c, T_go, T_bco')

a_torque_vector = a_torque*inertial_frame.z - b_torque*inertial_frame.z
a_link_torque = (a_frame, a_torque_vector)
b_torque_vector = b_torque*inertial_frame.z - c_torque*inertial_frame.z
b_link_torque = (b_frame, b_torque_vector)
c_torque_vector = c_torque*inertial_frame.z
c_link_torque = (c_frame, c_torque_vector)

# Equations of Motion
# ===================

coordinates = [theta_a, theta_b, theta_c]

speeds = [omega_a, omega_b, omega_c]

kane = KanesMethod(inertial_frame, coordinates, speeds, kinematical_differential_equations)

loads = [b_grav_force,
         c_grav_force,
         d_grav_force,
         a_link_torque,
         b_link_torque,
         c_link_torque]

bodies = [bP, cP, dP, goP, bcoP]

fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix
forcing = simplify(kane.forcing)
