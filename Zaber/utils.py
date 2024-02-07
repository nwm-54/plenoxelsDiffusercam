import cv2
import numpy as np
import time
import imageio.v3 as iio
import os
import gphoto2 as gp
import matplotlib.pyplot as plt
from skimage import filters, measure

def downsample_image(image, downsample_factor=2):
    new_width = image.shape[1] // downsample_factor
    new_height = image.shape[0] // downsample_factor
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def capture_image(camera):
    t1 = time.time()
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    target = os.path.join('/tmp', file_path.name)
    camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    img = iio.imread(target)
    os.remove(target)
    t2 = time.time()
    # print("Time took {} seconds".format(t2 - t1))
    return img

def process_image(image):
    grayscale_image = np.array(image)[:, :, 0]
    threshold_value = filters.threshold_otsu(grayscale_image)
    binary_image = grayscale_image > threshold_value
    labeled_image = measure.label(binary_image)
    properties = measure.regionprops(labeled_image)
    dot_pixel_locations = [(int(prop.centroid[1]), int(prop.centroid[0])) for prop in properties if prop.area > 1]
    return dot_pixel_locations