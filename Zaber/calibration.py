import gc
import pickle

import gphoto2 as gp
import numpy as np
from tqdm import tqdm
from zaber_motion import Units
from zaber_motion.ascii import Connection
import utils

serial_port = '/dev/tty.usbserial-AB0PG7GL'
serial_port = '/dev/ttyUSB0'

camera = gp.Camera()
camera.init()

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

    pixel_locations = {}
    try:
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

                    image = utils.downsample_image(utils.capture_image(), 8)  # Capture image at this position
                    dot_pixel_locations = utils.process_image(image)  # Process image to find dot locations

                    print("Detected {} dots. Locations: {}".format(len(dot_pixel_locations), dot_pixel_locations))
                    for pixel in dot_pixel_locations:
                        if pixel not in pixel_locations:
                            pixel_locations[pixel] = []
                        pixel_locations[pixel].append((x, y, z))

                    gc.collect()
    finally:
        camera.exit()

filename = 'pixel_locations.pkl'
with open(filename, 'wb') as f:
    pickle.dump(pixel_locations, f)
print(f"Saved pixel locations to {filename}")


