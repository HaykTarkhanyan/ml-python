import cv2
import numpy as np


# Checks if given directory is empty
def check_input(path):
    try:
        im = cv2.imread(path)
        im.shape
    except:
        print("Please check your input directory")
        return False
    return True


def convert_image(path):
    # read image, resize it and return valid numpy array
    im = cv2.imread(path)
    im.resize(28, 28, 1)
    im = img_to_array(im)
    im = np.array([im])
    return im
