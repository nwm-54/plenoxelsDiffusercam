import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

with open('pixel_locations.pkl', 'rb') as f:
    pixel_locations = pickle.load(f)
print("Loaded pixel locations from file")

# Visualization part
# pixel = (3325, 3370)  # Example pixel location to visualize
# print(len(list(pixel_locations.keys())))

# if pixel in pixel_locations:
#     xyz_values = np.array(pixel_locations[pixel])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(xyz_values[:,0], xyz_values[:,1], xyz_values[:,2], marker='o')

#     # Drawing lines between points (interpolation)
#     for i in range(len(xyz_values)-1):
#         ax.plot([xyz_values[i,0], xyz_values[i+1,0]], 
#                 [xyz_values[i,1], xyz_values[i+1,1]], 
#                 [xyz_values[i,2], xyz_values[i+1,2]], 'r')

#     plt.show()
# else:
#     print(f"No data for pixel {pixel}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each LED location for each pixel
for pixel, led_positions in pixel_locations.items():
    x_vals, y_vals, z_vals = zip(*led_positions)  # This creates separate lists of X, Y, and Z
    ax.scatter(x_vals, y_vals, z_vals, label=f'Pixel {pixel}')  # Plot all LED positions for this pixel

    # Optionally, connect the LED positions with lines
    if len(led_positions) > 1:  # Only draw lines if there are multiple points
        ax.plot(x_vals, y_vals, z_vals)

# Set labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the legend and plot
ax.legend()
plt.show()