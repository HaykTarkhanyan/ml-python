import os
import cv2
import argparse
import numpy as np
from utils import helpers
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

CLASSES = sorted(['facebook', 'twitter', 'whatsapp', 'linkedin', 'reddit'])
LOAD_MODEL_FROM = "weights_and_config"

def argument_parser():
    # get inputs from terminal
    parser = argparse.ArgumentParser()

    parser.add_argument('-load_model_from', type=str, required=False,
                        help="specify location of the model to be loaded",
                        default='pretrained_model')
    parser.add_argument('-input_image_dir', type=str, required=True,
                        help="specify path to data",
                        )
    
    args = parser.parse_args()
    return args

def load_model():
    # load json and create model
    json_file = open(os.path.join(LOAD_MODEL_FROM),  'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(LOAD_MODEL_FROM))
    print("Loaded model")

    return loaded_model

if __name__ == "__main__":
    # checks if input is valid
    if check_input(path):
        # parse arguments 
        args = argument_parser()
        path = args.input_image_dir
        # load model
        loaded_model = load_model(LOAD_MODEL_FROM)
        # convert image
        im = helpers.convert_image(path)
        # print prediction with higheset probability
        predic = loaded_model.predict(im)
        print(classes[np.array(predic).argmax()])

    else:
        print ("Failed to load the image")