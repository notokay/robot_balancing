from double_pendulum_setup import *
from sympy import Matrix, cos, sin

def getj_0j_1():
    a1, a2 = dynamicsymbols('a1, a2')

    mass_matrix = kane.mass_matrix
    forcing = kane.forcing
    tf_0 = Matrix([[cos(theta1), -sin(theta1), -leg_length*sin(theta1)], [sin(theta1), cos(theta1), leg_length*cos(theta1)], [0,0,1]])

    tf_1 = Matrix([[cos(theta2), -sin(theta2), -body_length*sin(theta2)], [sin(theta2), cos(theta2), body_length*cos(theta2)], [0,0,1]])
        
    tf_01 = simplify(tf_0*tf_1)
    
    com_0 = Matrix([-leg_length*sin(theta1), leg_length*cos(theta1), 1])
    
    com_1 = Matrix([-body_length*sin(theta2), body_length*cos(theta2), 1])
    
    com_01 = simplify(tf_0*com_1)
    
    com = Matrix(com_0 + com_01)
    
    j_0 = com_0.jacobian([theta1, theta2])
    j_1 = simplify(com_01.jacobian([theta1, theta2]))
    j = com.jacobian([theta1, theta2])
    
    q = Matrix(coordinates)
    qdot = Matrix(speeds)
    qddot = Matrix([a1, a2])
    return [j_0, j_1]
    
a1, a2 = dynamicsymbols('a1, a2')

mass_matrix = kane.mass_matrix
forcing = kane.forcing

tf_0 = Matrix([[cos(theta1), -sin(theta1), -leg_length*sin(theta1)], [sin(theta1), cos(theta1), leg_length*cos(theta1)], [0,0,1]])

tf_1 = Matrix([[cos(theta2), -sin(theta2), -body_length*sin(theta2)], [sin(theta2), cos(theta2), body_length*cos(theta2)], [0,0,1]])

tf_01 = simplify(tf_0*tf_1)

com_0 = Matrix([-1*leg_mass*leg_length*sin(theta1), leg_mass*leg_length*cos(theta1), 1])

com_1 = Matrix([-1*body_mass*body_length*sin(theta2), body_mass*body_length*cos(theta2), 1])

com_01 = simplify(tf_0*com_1)

com = Matrix(com_0 + com_01)

j_0 = com_0.jacobian([theta1, theta2])
j_1 = simplify(com_01.jacobian([theta1, theta2]))
j = com.jacobian([theta1, theta2])

q = Matrix(coordinates)
qdot = Matrix(speeds)
qddot = Matrix([a1, a2])
