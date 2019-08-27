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

def __init__():
    pass

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

def argument_parser_for_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('-load_model_from', type=str, required=False,
                        help="specify location of the model to be loaded",
                        default='pretrained_model')
    parser.add_argument('-input_image_dir', type=str, required=True,
                        help="specify path to data",
                        )


def argument_parser_for_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_save_dir', type=str, required=False,
                        help="specify location where model will be saved",
                        default='pretrained_model')
    parser.add_argument('-data_dir', type=str, required=False,
                        help="specify path to data",
                        default='data')
    parser.add_argument('-epochs', type=str, required=False,
                        help="specify number of epochs to train",
                        default='7')
    parser.add_argument('-batch_size', type=str, required=False,
                        help="specify batch_size",
                        default='1')

    args = parser.parse_args()
    return args

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








