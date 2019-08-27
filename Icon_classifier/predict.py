import os
import cv2
import argparse
import numpy as np
from utils import helpers
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

args = helpers.argument_parser_for_test()

CLASSES = sorted(['facebook', 'twitter', 'whatsapp', 'linkedin', 'reddit'])
LOAD_MODEL_FROM = args.load_model_from


# load json and create model
json_file = open(os.path.join(LOAD_MODEL_FROM),  'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(LOAD_MODEL_FROM))
print("Loaded model")



path = args.input_image_dir

if check_input(path):
    im = helpers.convert_image(path)
    # print prediction with higheset probability
    predic = loaded_model.predict(im)
    print(classes[np.array(predic).argmax()])

else:
    print ("Failed to load the image")