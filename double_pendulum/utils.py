#!/usr/bin/env python

import numpy as np
from numpy.linalg import det
from numpy import transpose


def controllable(a, b):
    """Returns true if the system is controllable and false if not.

    Parameters
    ----------
    a : array_like, shape(n,n)
        The state matrix.
    b : array_like, shape(n,r)
        The input matrix.

    Returns
    -------
    controllable : boolean

    """
    a = np.matrix(a)
    b = np.matrix(b)
    n = a.shape[0]
    controllability_matrix = []
    for i in range(n):
        controllability_matrix.append(a ** i * b)
    controllability_matrix = np.hstack(controllability_matrix)

    return np.linalg.matrix_rank(controllability_matrix) == n

def det_controllable(a, b):
    #Returns the determinant of the controllability matrix
    a = np.matrix(a)
    b = np.matrix(b)
    n = a.shape[0]
    controllability_matrix = []
    
    for i in range(n):
        controllability_matrix.append((a ** i) * b)
    return np.linalg.det(np.hstack(controllability_matrix))

def controllability_matrix(a,b):
    a = np.matrix(a)
    b = np.matrix(b)
    n = a.shape[0]
    controllability_matrix = []
    for i in range(n):
        print(a**i * b)
        controllability_matrix.append(a**i * b)
    return controllability_matrix

