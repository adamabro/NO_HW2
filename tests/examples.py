import numpy as np

def lp_example(point):
    obj_value = -(point[0] + point[1])
    grad = np.array([-1, -1])
    hess = np.zeros((2, 2))
    return obj_value, grad, hess

def lp_constraints():
    ineq_constraints = [
        (lambda point: point[0] - 2, lambda point: np.array([1, 0]), lambda point: np.zeros((2, 2))),
        (lambda point: -point[1], lambda point: np.array([0, -1]), lambda point: np.zeros((2, 2))),
        (lambda point: -point[1] + (-point[0] + 1), lambda point: np.array([-1, 1]), lambda point: np.zeros((2, 2))),
        (lambda point: point[1] - 1, lambda point: np.array([0, 1]), lambda point: np.zeros((2, 2)))
    ]
    eq_mat = np.array([])  
    eq_rhs = np.array([])
    return ineq_constraints, eq_mat, eq_rhs

def qp_example(point):
    obj_value = point[0]**2 + point[1]**2 + (point[2] + 1)**2
    grad = np.array([2*point[0], 2*point[1], 2*(point[2] + 1)])
    hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return obj_value, grad, hess

def qp_constraints():
    ineq_constraints = [
        (lambda point: -point[0], lambda point: np.array([-1, 0, 0]), lambda point: np.zeros((3, 3))),
        (lambda point: -point[1], lambda point: np.array([0, -1, 0]), lambda point: np.zeros((3, 3))),
        (lambda point: -point[2], lambda point: np.array([0, 0, -1]), lambda point: np.zeros((3, 3)))
    ]
    eq_mat = np.array([[1, 1, 1]])
    eq_rhs = np.array([1])
    return ineq_constraints, eq_mat, eq_rhs
