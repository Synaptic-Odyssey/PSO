import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#This is a problem set for programming club members!

# Define a distinct, non-random objective function
def objective_function(x, y):
    return np.sin(x) * np.cos(y) + 0.1 * (x - 5)**2 + 0.1 * (y - 5)**2

# Create meshgrid for visualization
x_lin = np.linspace(0, 10, 100)
y_lin = np.linspace(0, 10, 100)
x_vals, y_vals = np.meshgrid(x_lin, y_lin)
z_vals = objective_function(x_vals, y_vals)

# Compute the actual (approximate) global minimum from the grid
min_idx = np.argmin(z_vals)
min_idx = np.unravel_index(min_idx, z_vals.shape)
actual_min_x = x_vals[min_idx]
actual_min_y = y_vals[min_idx]
actual_min_value = z_vals[min_idx]

# PSO Hyperparameters
#TODO: Fix these parameters!
c1, c2, w = 0.00000001, 0.2, 0.897      

num_particles = 5
np.random.seed(100)
positions = np.random.rand(2, num_particles) * 4
velocities = np.random.randn(2, num_particles) * 20


# Initialize best positions and values
personal_best_positions = positions.copy()
personal_best_values = objective_function(personal_best_positions[0], personal_best_positions[1])
global_best_index = personal_best_values.argmin()
global_best_position = personal_best_positions[:, global_best_index]
global_best_value = personal_best_values.min()

def update_particles():
    global velocities, positions, personal_best_positions, personal_best_values, global_best_position, global_best_value
    
    r1, r2 = np.random.rand(2)
    velocities = w * velocities + c1 * r1 * (personal_best_positions - positions) + \
                 c2 * r2 * (global_best_position.reshape(-1, 1) - positions)
    positions += velocities

    # Evaluate new positions
    current_values = objective_function(positions[0], positions[1])

    # Update personal best
    better_mask = current_values < personal_best_values
    personal_best_positions[:, better_mask] = positions[:, better_mask]
    personal_best_values[better_mask] = current_values[better_mask]

    # Update global best
    new_global_best_index = personal_best_values.argmin()
    global_best_position = personal_best_positions[:, new_global_best_index]
    global_best_value = personal_best_values.min()

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_tight_layout(True)
img = ax.imshow(z_vals, extent=[0, 10, 0, 10], origin='lower', cmap='plasma', alpha=0.7)
fig.colorbar(img, ax=ax)

# Draw a triangle marker at the actual (approximate) global minimum
ax.plot(actual_min_x, actual_min_y, marker='^', markersize=10, color="white", label="Actual Min")

# Contour plot for better visualization
contours = ax.contour(x_vals, y_vals, z_vals, 10, colors='purple', alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f", colors='white')

# Plot elements for PSO particles
pbest_plot = ax.scatter(personal_best_positions[0], personal_best_positions[1], marker='o', color='lime', alpha=0.6)
gbest_plot = ax.scatter(global_best_position[0], global_best_position[1], marker='s', s=120, color='yellow', alpha=0.7)

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.legend()

def animate(i):
    update_particles()
    ax.set_title(f"Step: {i+1}")
    pbest_plot.set_offsets(personal_best_positions.T)
    gbest_plot.set_offsets(global_best_position.reshape(1, -1))
    return ax, pbest_plot, gbest_plot

#TODO: optimize the number of iterations!
anim = FuncAnimation(fig, animate, frames=50, interval=500, blit=False, repeat=True)
plt.show()

difference = abs(global_best_value - actual_min_value)
print("PSO algorithm converged (but probably not to the true minimum due to scrambled coefficients).")
print(f"Best solution found at coordinates: {global_best_position} with objective function value: {global_best_value}")
print(f"The (approximate) global minimum is located at: ({actual_min_x}, {actual_min_y}) with a value of {actual_min_value}")
print(f"Difference between PSO-found minimum and actual minimum: {difference}")
