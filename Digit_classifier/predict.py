import os
import cv2
import argparse
import numpy as np
from check_input import check_input
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

LOAD_MODEL_FROM = 'ckpt'

# load json and create model
json_file = open(os.path.join(LOAD_MODEL_FROM, 'model_cnn.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(LOAD_MODEL_FROM, 'model_cnn.h5'))
print("Loaded model from ckpt folder")


parser = argparse.ArgumentParser()

parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

args = parser.parse_args()
path = args.inp_dir

# before trying to predict making sure that directory is valid
if check_input(path):
    im = cv2.imread(path)
    im.resize(28, 28, 1)
    im = img_to_array(im)
    im = np.array([im])

    print(loaded_model.predict(im).argmax())

else:
    check_input(path)
