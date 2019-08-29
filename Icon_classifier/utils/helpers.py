import cv2
import keras
import argparse
import numpy as np
from keras import Sequential
from numpy import expand_dims
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# Checks if given directory is empty
def check_input(path):
    try:
        im = cv2.imread(path)
        im.shape
    except:
        print("Please check your input directory")
        return False
    return True


def convert_image(im):
    # read image, resize it and return valid numpy array
    im = cv2.imread(path)
    im.resize(25, 25, 3)
    im = img_to_array(im)
    im = np.array([im])
    return im

def load_data(LOAD_DATA_FROM):
    # importing data
    data_get = ImageDataGenerator()

    datagen = ImageDataGenerator(validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        LOAD_DATA_FROM,
        target_size=(25, 25),
        batch_size=8,
    )

    valid = datagen.flow_from_directory(
        LOAD_DATA_FROM,
        target_size=(25, 25),
    )

    return train_generator, valid








