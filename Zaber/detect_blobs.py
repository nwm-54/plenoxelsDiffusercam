import cv2
import matplotlib.pyplot as plt
import utils


# Process the image to find the dots
img = utils.downsample_image(utils.capture_image(), 8)

dot_pixel_locations = utils.process_image(img)

# The number of detected dots should be printed out
print(len(dot_pixel_locations), dot_pixel_locations)
image_with_dots = img.copy()

# Draw a green circle around each dot
for (x, y) in dot_pixel_locations:
    cv2.circle(image_with_dots, (x, y), 20, (0, 255, 0), 4)

plt.imshow(image_with_dots)
plt.show()
