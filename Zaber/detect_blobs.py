import cv2
import numpy as np
import time
import imageio.v3 as iio
import os
import gphoto2 as gp
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, measure

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
    target = os.path.join('/Users/shamus/Downloads/', file_path.name)
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


# Process the image to find the dots
img = downsample_image(capture_image(), 8)

dot_pixel_locations = process_image(img)

# The number of detected dots should be printed out
print(len(dot_pixel_locations), dot_pixel_locations)
image_with_dots = img.copy()

# Draw a green circle around each dot
for (x, y) in dot_pixel_locations:
    cv2.circle(image_with_dots, (x, y), 20, (0, 255, 0), 4)

plt.imshow(image_with_dots)
plt.show()
