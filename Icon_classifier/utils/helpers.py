import cv2
import keras
import argparse
import numpy as np
from keras import Sequential
from numpy import expand_dims
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import model_from_json
from keras.layers import Conv3D, MaxPooling3D
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


def load_model(folder):
    # load json and create model
    json_file = open(os.path.join(folder, 'model_cnn.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(folder, 'model_cnn.h5'))
    print("Loaded model from ckpt folder")
    return loaded_mode

def save_model(folder):        
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(folder, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(folder, "model.h5"))
    print("Saved model to ckpt folder")


def convert_image(im):
    # read image, resize it and return valid numpy array
    im = cv2.imread(path)
    im.resize(25, 25, 3)
    im = img_to_array(im)
    im = np.array([im])
    return im


def argument_parser():
    # takes input directory via terminal
    parser = argparse.ArgumentParser()

    parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

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

def keras_model(num_classes):
    # Network that does all the job
    classifier = Sequential()

    classifier.add(Dense(100, activation='relu',
                        input_shape=train_generator.image_shape))
    classifier.add(Flatten())
    classifier.add(Dense(80, activation='relu'))
    classifier.add(Dense(80, activation='relu'))
    classifier.add(Dense(num_classes,  activation='softmax'))

    nadam = keras.optimizers.nadam(lr=.000000753)

    classifier.compile(
        optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier









