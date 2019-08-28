import os
import cv2
import argparse
import numpy as np
from utils import helpers
from check_input import check_input
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# # first variant
# # can be overwritten via argparser
# LOAD_MODEL_WEIGHTS = "weights_and_config/model_cnn.h5"
# LOAD_MODEL_CONFIG = "weights_and_config/model_cnn.json"


# parser = argparse.ArgumentParser()

# parser.add_argument('-weights', type=str, required=False,
#                     help="specify location of the models weights to be loaded",
#                     default='weights_and_config/model_cnn.h5')

# parser.add_argument('-config', type=str, required=False,
#                     help="specify location of the models config file to be loaded",
#                     default='weights_and_config/model_cnn.json')

# parser.add_argument('-input_image_dir', type=str, required=True,
#                     help="specify path to data",
#                     )

# args = parser.parse_args()

# LOAD_MODEL_WEIGHTS = args.weights
# LOAD_MODEL_CONFIG = args.config

#################################################################################
# second variant

# parser = argparse.ArgumentParser()

# parser.add_argument('-weights', type=str, required=False,
#                     help="specify location of the models weights to be loaded",
#                     default='weights_and_config/model_cnn.h5')

# parser.add_argument('-config', type=str, required=False,
#                     help="specify location of the models config file to be loaded",
#                     default='weights_and_config/model_cnn.json')

# parser.add_argument('-input_image_dir', type=str, required=True,
#                     help="specify path to data",
#                     )

# args = parser.parse_args()

# LOAD_MODEL_WEIGHTS = args.weights
# LOAD_MODEL_CONFIG = args.config


def load_model(weights, config):    
    # load json and create model
    json_file = open(os.path.join(config), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(weights))
    print("Loaded model")

    return loaded_model


path = args.input_image_dir

if __name__ == "__main__":
    # load model
    loaded_model = load_model(LOAD_MODEL_WEIGHTS, LOAD_MODEL_CONFIG)
    # before trying to predict making sure that directory is valid
    if check_input(path):
        im = helpers.convert_image(path)
        # print the prediction based on probabilities
        print(loaded_model.predict(im).argmax())

    else:
        print ("Failed to load image")










