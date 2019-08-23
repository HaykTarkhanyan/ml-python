import cv2
import argparse
from check_input import check_input
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import os

classes = sorted(['facebook', 'twitter', 'whatsapp', 'linkedin', 'reddit'])

# load json and create model
json_file = open(os.path.join('ckpt', 'model.json'),  'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join("ckpt", "model.h5"))
print("Loaded model")


parser = argparse.ArgumentParser()

parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

args = parser.parse_args()
path = args.inp_dir

if check_input(path):
    im = cv2.imread(path)
    im.resize(25, 25, 3)
    im = img_to_array(im)
    im = np.array([im])

    predic = loaded_model.predict(im)
    print()
    print(classes[np.array(predic).argmax()])

else:
    check_input(path)
