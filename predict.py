import cv2
import argparse
import numpy as np
import os
from check_input import check_input

from keras.preprocessing.image import img_to_array
from keras.models import model_from_json

# load json and create model
json_file = open(os.path.join('ckpt', 'model_cnn.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join('ckpt', 'model_cnn.h5'))
print("Loaded model from ckpt folder")


parser = argparse.ArgumentParser()

parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

args = parser.parse_args()
path = args.inp_dir

if check_input(path):
    im = cv2.imread(path)
    im.resize(28, 28, 1)
    im = img_to_array(im)
    im = np.array([im])

    print(loaded_model.predict(im).argmax())

else:
    check_input(path)
