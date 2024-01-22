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
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use Gaussian blur to reduce noise and improve dot detection
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    
    # Threshold the image to create a binary image where the dots are white
    _, binary_image = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary_image)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Calculate the weighted average position of the contours
    dot_pixel_locations = []
    for contour in contours:
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dot_pixel_locations.append((cX, cY))
    
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

    z_range = np.arange(31.5, 32.5, 0.2)
    x_range = np.arange(20, 29.1, 0.5)
    y_range = np.arange(17.5, 24, 0.5)

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

                image = capture_image()  # Capture image at this position
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


