import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

with open('pixel_locations.pkl', 'rb') as f:
    pixel_locations = pickle.load(f)
print("Loaded pixel locations from file")

# Visualization part
pixel = (10, 20)  # Example pixel location to visualize

if pixel in pixel_locations:
    xyz_values = np.array(pixel_locations[pixel])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_values[:,0], xyz_values[:,1], xyz_values[:,2], marker='o')

    # Drawing lines between points (interpolation)
    for i in range(len(xyz_values)-1):
        ax.plot([xyz_values[i,0], xyz_values[i+1,0]], 
                [xyz_values[i,1], xyz_values[i+1,1]], 
                [xyz_values[i,2], xyz_values[i+1,2]], 'r')

    plt.show()
else:
    print(f"No data for pixel {pixel}")