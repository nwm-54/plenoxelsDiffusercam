import os
import time

import gphoto2 as gp

from zaber_motion import Units
from zaber_motion.ascii import Connection
import imageio.v3 as iio
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from skimage import measure, filters

def downsample_image(image, downsample_factor=2):
    """
    Downsample the image by the provided factor.
    """
    # Calculate the new dimensions
    new_width = image.shape[1] // downsample_factor
    new_height = image.shape[0] // downsample_factor
    
    # Downsample the image
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return downsampled_image

# Load the image
def capture_image():
    t1 = time.time()
    camera = gp.Camera()
    camera.init()
    # print('Capturing image')
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    # print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('/tmp', file_path.name)
    # print('Copying image to', target)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    img = iio.imread(target)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    camera.exit()
    t2 = time.time()
    # print("Time took {} seconds".format(t2 - t1))
    return img

def process_image(image):
    grayscale_image = np.array(image)[:, :, 0]

    # Apply Otsu's threshold to the image
    threshold_value = filters.threshold_otsu(grayscale_image)

    # Generate the binary image
    binary_image = grayscale_image > threshold_value

    # Label the image
    labeled_image = measure.label(binary_image)
    # Compute the properties of the labeled image
    properties = measure.regionprops(labeled_image)

    # Get the coordinates of the centroids of the labeled regions
    dot_pixel_locations = [(int(prop.centroid[1]), int(prop.centroid[0])) 
                           for prop in properties if prop.area > 1]
    return dot_pixel_locations

serial_port = '/dev/tty.usbserial-AB0PG7GL'

with Connection.open_serial_port(serial_port) as connection:
    connection.enable_alerts()
    device_list = connection.detect_devices()
    print(f"Found {len(device_list)} devices")
    if len(device_list) != 3:
        raise ValueError("Did not find all devices.")

    for device in device_list:
        print(f"Homing all axes of device with address {device.device_address}.")
        device.all_axes.home()

    z_range = np.arange(31.9, 33.6, 0.2)
    x_range = np.arange(20, 29.1, 0.5)
    y_range = np.arange(18.0, 24, 0.5)

    z_axis, x_axis, y_axis = device_list[0].get_axis(1), device_list[1].get_axis(1), device_list[2].get_axis(1)

    pixel_locations = {}  # Dictionary to store pixel locations for each XYZ coordinate

    for z in tqdm(z_range):
        z_axis.move_absolute(z, Units.LENGTH_MILLIMETRES)
        z_axis.wait_until_idle() 

        for y in y_range:
            y_axis.move_absolute(y, Units.LENGTH_MILLIMETRES)
            y_axis.wait_until_idle()
            
            x_range_current = x_range if y % 2 == 0 else reversed(x_range)

            for x in x_range_current:
                x_axis.move_absolute(x, Units.LENGTH_MILLIMETRES)
                x_axis.wait_until_idle()
                print(f"Moved to point: X={x} mm, Y={y} mm, Z={z} mm")

                image = downsample_image(capture_image(), 8)  # Capture image at this position
                dot_pixel_locations = process_image(image)  # Process image to find dot locations

                print("Detected {} dots. Locations: {}".format(len(dot_pixel_locations), dot_pixel_locations))
                for pixel in dot_pixel_locations:
                    if pixel not in pixel_locations:
                        pixel_locations[pixel] = []
                    pixel_locations[pixel].append((x, y, z))


filename = 'pixel_locations.pkl'
with open(filename, 'wb') as f:
    pickle.dump(pixel_locations, f)
print(f"Saved pixel locations to {filename}")


