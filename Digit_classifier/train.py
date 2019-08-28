import os
import cv2
import keras
import argparse
import numpy as np
import pandas as pd
from utils import helpers
from CNN import keras_model
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array


# PATH_TO_SAVE_WEIGHTS = os.path.join('weights_and_config','model.h5')
# PATH_TO_SAVE_CONFIG = os.path.join('weights_and_config', 'model.json')

# PATH_TO_LOAD_DATA = os.path.join('data', 'train.csv')

# EPOCHS = 7
# BATCH_SIZE = 1
# NUM_CLASSES = 10

# takes data location, path where to save models weights, config 
# num of epochs and batch size
parser = argparse.ArgumentParser()

parser.add_argument('-save_weights', type=str, required=False,
                    help="specify location where model will be saved",
                    default=os.path.join('weights_and_config','model.h5'))
parser.add_argument('-save_config', type=str, required=False,
                    help="specify location where model will be saved",
                    default=os.path.join('weights_and_config', 'model.json'))
parser.add_argument('-data_dir', type=str, required=False,
                    help="specify path to data",
                    default='data')
parser.add_argument('-epochs', type=str, required=False,
                    help="specify number of epochs to train",
                    default=7)
parser.add_argument('-batch_size', type=str, required=False,
                    help="specify batch_size",
                    default=1)

args = parser.parse_args()



PATH_TO_SAVE_WEIGHTS = args.model_save_weights
PATH_TO_SAVE_CONFIG = args.model_save_config

PATH_TO_LOAD_DATA = args.data_dir

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_CLASSES = 10


def load_data(path)
    # load data from csv
    data = pd.read_csv(os.path.join(path))
    # test = pd.read_csv(os.path.join(PATH_TO_LOAD_DATA))

    # get image labels
    y_train = data["label"]
    del data['label']

    x_train = data.to_numpy()
    x_train = np.array([i.reshape((28, 28, 1)) for i in x_train])
    # convert labels to vector of zeros and one
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train

def train_model():
    model = keras_model(NUM_CLASSES)

    model.compile(optimizer='AdaDelta',
                loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        )
    
    return model, history

def save_model(model,path_to_save_weights, path_to_save_config):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path_to_save_config), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path_to_save_weights))
    print("Saved model")

if __name__ == __main__:
    # load data
    x_train, y_train = load_data(PATH_TO_LOAD_DATA)
    # train model
    model, history = train_model()
    # save model
    save_model(model, PATH_TO_SAVE_WEIGHTS, PATH_TO_SAVE_CONFIG)





