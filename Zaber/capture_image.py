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

if __name__ == "__main__":
    capture_image()