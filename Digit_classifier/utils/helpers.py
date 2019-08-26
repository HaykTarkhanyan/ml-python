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

def argument_parser():
    # takes input directory via terminal
    parser = argparse.ArgumentParser()

    parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

    args = parser.parse_args()
    return args

def convert_image(im):
    # read image, resize it and return valid numpy array
    im = cv2.imread(path)
    im.resize(28, 28, 1)
    im = img_to_array(im)
    im = np.array([im])
    return im

def keras_model(num_classes):
    # model taken from keras documentation
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    ))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='AdaDelta',
                loss="categorical_crossentropy", metrics=["accuracy"])

    return model