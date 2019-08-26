import os
import cv2
import argparse
import numpy as np
from utils import helpers
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, Dropout, MaxPooling2D


def __init__():
    # helps to use this file in another one
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

def load_model(folder):
    # load json and create model
    json_file = open(os.path.join(folder, 'model_cnn.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(folder, 'model_cnn.h5'))
    print("Loaded model from ckpt folder")
    return loaded_model

def save_model(folder):        
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(folder, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(folder, "model.h5"))
    print("Saved model to ckpt folder")

def load_data(folder):
    # load data from csv
    data = pd.read_csv(os.path.join(folder, 'train.csv'))
    test = pd.read_csv(os.path.join(folder, 'test.csv'))

    # get image labels
    y_train = data["label"]
    del data['label']

    x_train = data.to_numpy()
    x_train = np.array([i.reshape((28, 28, 1)) for i in x_train])

    # convert labels to vector of zeros and one
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train, test

def argument_parser_for_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_save_dir', type=str, required=False,
                        help="specify location where model will be saved",
                        default='ckpt')
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


def argument_parser_for_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('-load_model_from', type=str, required=False,
                        help="specify location of the model to be loaded",
                        default='ckpt')
    parser.add_argument('-input_image_dir', type=str, required=True,
                        help="specify path to data",
                        )


    args = parser.parse_args()
    return args


def convert_image(path):
    # read image, resize it and return valid numpy array
    im = cv2.imread(path)
    im.resize(28, 28, 1)
    im = img_to_array(im)
    im = np.array([im])
    return im
