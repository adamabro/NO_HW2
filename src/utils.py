import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_objective_contours(objective_function, x_limits, y_limits, paths=None):
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_values = np.linspace(x_min, x_max, 100)
    y_values = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.array([[objective_function(np.array([x_val, y_val])) for x_val in x_values] for y_val in y_values])
    
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    
    if paths:
        for path, label in paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=label, marker='o', color='blue')
    
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Objective Contours')
    plt.colorbar(label='Objective Function Value')
    plt.grid(True)
    plt.show()

def draw_function_values(*data):
    for values, label in data:
        if values:
            plt.plot(values, label=label, marker='o', color='red')
    
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Function Value')
    plt.title('Function Values Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_qp_path(path):
    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='green')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title("Quadratic Programming")
    plt.grid(True)
    plt.show()

def draw_lp_path(path):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], marker='o', color='purple')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("Linear Programming")
    plt.grid(True)
    plt.show()

def plot_final_candidate(final_candidate, problem_name):
    plt.figure()
    plt.scatter(final_candidate[0], final_candidate[1], color='red')
    plt.title(f'Final Candidate for {problem_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig(f'{problem_name}_final_candidate.png')
    plt.show()

def plot_objective_constraint(final_candidate, objective_value, constraint_values, problem_name):
    plt.figure()
    plt.bar(['Objective'] + [f'Constraint {i+1}' for i in range(len(constraint_values))],
            [objective_value] + constraint_values)
    plt.title(f'Objective and Constraints at Final Candidate for {problem_name}')
    plt.ylabel('Value')
    plt.grid()
    plt.savefig(f'{problem_name}_objective_constraints.png')
    plt.show()

def generate_feasible_region(constraints, x_limits, y_limits, resolution=100):
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.ones_like(X)
    
    for constraint in constraints:
        Z = np.logical_and(Z, constraint(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape) <= 0)
    
    return X, Y, Z

def plot_feasible_region_path(constraints, path, problem_name, x_limits, y_limits):
    X, Y, Z = generate_feasible_region(constraints, x_limits, y_limits)
    
    plt.figure()
    plt.contourf(X, Y, Z, levels=[0, 1], colors=['#539ecd', '#e8f7ff'])
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'b--', marker='o')
    plt.title(f'Feasible Region and Path for {problem_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig(f'{problem_name}_feasible_region_path.png')
    plt.show()

def plot_objective_vs_iteration(objective_values, problem_name):
    plt.figure()
    plt.plot(objective_values, 'b-o')
    plt.title(f'Objective Value vs Iteration Number for {problem_name}')
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Value')
    plt.grid()
    plt.savefig(f'{problem_name}_objective_vs_iteration.png')
    plt.show()
