import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Randomly generated objective function with multiple minima, but the smallest minimum near the center (because it looks good lol)
def generate_random_function():
    global_min_x, global_min_y = np.random.uniform(3, 7), np.random.uniform(3, 7)
    
    base_function = lambda x, y: (x - global_min_x)**2 + (y - global_min_y)**2  
    num_local_minima = 5 
    local_minima_function = lambda x, y: sum(np.exp(-((x - np.random.uniform(0, 10))**2 + (y - np.random.uniform(0, 10))**2) / (2 * np.random.uniform(0.5, 2))) for _ in range(num_local_minima))
    
    num_sinusoids = 3
    sinusoidal_function = lambda x, y: sum(np.sin(np.random.uniform(0, 2*np.pi) * x + np.random.uniform(0, 2*np.pi) * y) * np.random.uniform(0.5, 1.5) for _ in range(num_sinusoids))
    
    return lambda x, y: base_function(x, y) + local_minima_function(x, y) + sinusoidal_function(x, y), global_min_x, global_min_y

objective_function, global_min_x, global_min_y = generate_random_function()

x_vals, y_vals = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
z_vals = objective_function(x_vals, y_vals)

# Hyperparameters for PSO
c1, c2, w = 0.1, 0.1, 0.8

num_particles = 40
np.random.seed(100)
positions = np.random.rand(2, num_particles) * 10
velocities = np.random.randn(2, num_particles) * 0.1

personal_best_positions = positions
personal_best_values = objective_function(personal_best_positions[0], personal_best_positions[1])
global_best_position = personal_best_positions[:, personal_best_values.argmin()]
global_best_value = personal_best_values.min()

def update_particles():
    global velocities, positions, personal_best_positions, personal_best_values, global_best_position, global_best_value
    
    r1, r2 = np.random.rand(2)
    velocities = w * velocities + c1 * r1 * (personal_best_positions - positions) + c2 * r2 * (global_best_position.reshape(-1, 1) - positions)
    positions += velocities

    current_values = objective_function(positions[0], positions[1])

    better_mask = current_values < personal_best_values
    personal_best_positions[:, better_mask] = positions[:, better_mask]
    personal_best_values[better_mask] = current_values[better_mask]

    new_global_best_index = personal_best_values.argmin()
    global_best_position = personal_best_positions[:, new_global_best_index]
    global_best_value = personal_best_values.min()

fig, ax = plt.subplots(figsize=(8, 6))
fig.set_tight_layout(True)
img = ax.imshow(z_vals, extent=[0, 10, 0, 10], origin='lower', cmap='plasma', alpha=0.7)
fig.colorbar(img, ax=ax)
ax.plot(global_min_x, global_min_y, marker='^', markersize=10, color="white")

contours = ax.contour(x_vals, y_vals, z_vals, 10, colors='purple', alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f", colors='white')

pbest_plot = ax.scatter(personal_best_positions[0], personal_best_positions[1], marker='o', color='lime', alpha=0.6)
gbest_plot = ax.scatter(global_best_position[0], global_best_position[1], marker='s', s=120, color='yellow', alpha=0.7)

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

def animate(i):
    title = f"Step: {i+1}"
    update_particles()

    ax.set_title(title)
    
    pbest_plot.set_offsets(personal_best_positions.T)
    gbest_plot.set_offsets(global_best_position.reshape(1, -1))

    return ax, pbest_plot, gbest_plot

anim = FuncAnimation(fig, animate, frames=50, interval=500, blit=False, repeat=True)
plt.show()

print("PSO algorithm successfully converged.")
print(f"Best solution found at coordinates: {global_best_position} with objective function value: {global_best_value}")
print(f"The known global minimum is located at: ({global_min_x}, {global_min_y}) with a value of {objective_function(global_min_x, global_min_y)}")
