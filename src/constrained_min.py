import numpy as np

def perform_newton_step(func, grad_func, hess_func, point, constraint_funcs, eq_mat, eq_rhs, barrier_param, iter_count):
    def lagrangian_grad(point):
        grad_lagrangian = barrier_param * grad_func(point) + log_barrier_gradient(point, constraint_funcs)
        if eq_mat is not None and eq_mat.size > 0:
            grad_lagrangian += eq_mat.T @ np.linalg.solve(eq_mat @ eq_mat.T, eq_rhs - eq_mat @ point)
        return grad_lagrangian

    def lagrangian_hess(point):
        hess_lagrangian = barrier_param * hess_func(point) + log_barrier_hessian(point, constraint_funcs)
        if eq_mat is not None and eq_mat.size > 0:
            hess_lagrangian += eq_mat.T @ np.linalg.solve(eq_mat @ eq_mat.T, np.eye(eq_mat.shape[0])) @ eq_mat
        return hess_lagrangian

    grad = lagrangian_grad(point).astype(np.float64)
    hess = lagrangian_hess(point).astype(np.float64)
    if not np.isfinite(grad).all() or not np.isfinite(hess).all():
        return point
    direction = -np.linalg.solve(hess, grad)
    step_size = backtrack_line_search(lambda point: barrier_param * func(point) + log_barrier_term(point, constraint_funcs), lagrangian_grad, point, direction, constraint_funcs=constraint_funcs)
    print(f"Iteration {iter_count}: Objective function value = {func(point)}")
    return point + step_size * direction

def backtrack_line_search(func, grad_func, point, direction, step_size=1.0, reduction_factor=0.8, condition_param=1e-4, constraint_funcs=None):
    while (func(point + step_size * direction) > func(point) + condition_param * step_size * np.dot(grad_func(point), direction)) or \
          (constraint_funcs and any(constraint(point + step_size * direction) >= 0 for constraint, _, _ in constraint_funcs)):
        if step_size < 1e-10:
            break
        step_size *= reduction_factor
    return step_size

def log_barrier_term(point, constraint_funcs):
    penalties = [-np.log(-constraint(point)) for constraint, _, _ in constraint_funcs if constraint(point) < 0]
    penalty = -np.sum(penalties)
    return penalty if penalty != np.inf else np.inf

def log_barrier_gradient(point, constraint_funcs):
    gradient = np.zeros_like(point, dtype=np.float64)
    for constraint, grad_constraint, _ in constraint_funcs:
        if constraint(point) >= 0:
            return np.inf * np.ones_like(point, dtype=np.float64)
        gradient += grad_constraint(point) / -constraint(point)
    return gradient

def log_barrier_hessian(point, constraint_funcs):
    hessian = np.zeros((len(point), len(point)), dtype=np.float64)
    for constraint, grad_constraint, hess_constraint in constraint_funcs:
        if constraint(point) >= 0:
            return np.inf * np.ones((len(point), len(point)), dtype=np.float64)
        grad_outer = grad_constraint(point).reshape(-1, 1) @ grad_constraint(point).reshape(1, -1)
        hessian += grad_outer / constraint(point)**2 - hess_constraint(point)
    return hessian

def interior_point_method(func, grad_func, hess_func, constraint_funcs, eq_mat, eq_rhs, initial_point, barrier_param=1, barrier_increase=10, obj_tol=1e-12, param_tol=1e-8, func_tol=1e-12, max_iters=100):
    point = np.array(initial_point, dtype=np.float64)
    points_path = [point.copy()]
    iter_count = 0
    prev_func_val = func(point)

    if any(constraint(point) >= 0 for constraint, _, _ in constraint_funcs):
        raise ValueError("Initial point is not feasible")

    for _ in range(max_iters):
        prev_point = point.copy()
        for _ in range(20):
            iter_count += 1
            point = perform_newton_step(func, grad_func, hess_func, point, constraint_funcs, eq_mat, eq_rhs, barrier_param, iter_count)
            current_func_val = func(point)
            if np.linalg.norm(point - prev_point) < param_tol or abs(current_func_val - prev_func_val) < func_tol:
                break
            prev_func_val = current_func_val
            prev_point = point.copy()
        points_path.append(point.copy())
        barrier_param = min(barrier_param * barrier_increase, 1e10)
        if np.linalg.norm(grad_func(point)) < obj_tol:
            break

    return point, func(point), True, points_path

class ConstrainedOptimizer:
    def __init__(self, func, grad_func, hess_func, constraint_funcs, eq_mat, eq_rhs):
        self.func = func
        self.grad_func = grad_func
        self.hess_func = hess_func
        self.constraint_funcs = constraint_funcs
        self.eq_mat = eq_mat
        self.eq_rhs = eq_rhs

    def optimize(self, initial_point, obj_tol, param_tol=1e-8, func_tol=1e-10, max_iters=100):
        return interior_point_method(self.func, self.grad_func, self.hess_func, self.constraint_funcs, self.eq_mat, self.eq_rhs, initial_point, obj_tol=obj_tol, param_tol=param_tol, func_tol=func_tol, max_iters=max_iters)
