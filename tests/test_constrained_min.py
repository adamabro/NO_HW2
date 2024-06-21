import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from constrained_min import *
from utils import *
from examples import *

class TestConstrainedOptimization(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-6
        self.param_tol = 1e-4
        self.max_iter = 100
        
        self.init_lp_point = [0.5, 0.75]
        self.init_qp_point = [0.1, 0.2, 0.7]
        
    def test_lp(self):
        lp_obj_func = lambda point: lp_example(point)[0]
        lp_grad_func = lambda point: lp_example(point)[1]
        lp_hess_func = lambda point: lp_example(point)[2]
        lp_ineq_constraints, lp_eq_mat, lp_eq_rhs = lp_constraints()

        lp_optimizer = ConstrainedOptimizer(lp_obj_func, lp_grad_func, lp_hess_func, lp_ineq_constraints, lp_eq_mat, lp_eq_rhs)
        lp_solution, lp_obj_value, lp_success, lp_path = lp_optimizer.optimize(self.init_lp_point, self.obj_tol, self.param_tol, self.max_iter)

        self.assertTrue(lp_success, "Optimization failed for linear programming")
        self.assertTrue(np.all(np.array([lp_solution[0] <= 2, lp_solution[1] >= -lp_solution[0] + 1, lp_solution[1] <= 1, lp_solution[1] >= 0])), "constraints violated")

        draw_lp_path(lp_path)
        
        # Plotting additional functionalities
        plot_final_candidate(lp_solution, 'Test_LP')
        plot_objective_constraint(lp_solution, lp_obj_value, [constraint(lp_solution) for constraint in lp_ineq_constraints], 'Test_LP')
        x_limits = (-10, 10)
        y_limits = (-10, 10)
        plot_feasible_region_path(lp_ineq_constraints, lp_path, 'Test_LP', x_limits, y_limits)
        plot_objective_vs_iteration([lp_obj_func(point) for point in lp_path], 'Test_LP')
        
    def test_qp(self):
        qp_obj_func = lambda point: qp_example(point)[0]
        qp_grad_func = lambda point: qp_example(point)[1]
        qp_hess_func = lambda point: qp_example(point)[2]
        qp_ineq_constraints, qp_eq_mat, qp_eq_rhs = qp_constraints()

        qp_optimizer = ConstrainedOptimizer(qp_obj_func, qp_grad_func, qp_hess_func, qp_ineq_constraints, qp_eq_mat, qp_eq_rhs)
        qp_solution, qp_obj_value, qp_success, qp_path = qp_optimizer.optimize(self.init_qp_point, self.obj_tol, self.param_tol, self.max_iter)

        self.assertTrue(qp_success, "Optimization failed for quadratic programming")
        self.assertTrue(np.all(np.array(qp_solution) >= 0), "constraints violated")

        draw_qp_path(qp_path)
        
        # Plotting additional functionalities
        plot_final_candidate(qp_solution, 'Test_QP')
        plot_objective_constraint(qp_solution, qp_obj_value, [constraint(qp_solution) for constraint in qp_ineq_constraints], 'Test_QP')
        x_limits = (-10, 10)
        y_limits = (-10, 10)
        plot_feasible_region_path(qp_ineq_constraints, qp_path, 'Test_QP', x_limits, y_limits)
        plot_objective_vs_iteration([qp_obj_func(point) for point in qp_path], 'Test_QP')

if __name__ == '__main__':
    unittest.main()
