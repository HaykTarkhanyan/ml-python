import numpy as np
from utils import helpers

CLASSES = sorted(['facebook', 'twitter', 'whatsapp', 'linkedin', 'reddit'])

# loading the model
LOAD_MODEL_FROM = 'ckpt'
loaded_model = helpers.load_model(LOAD_MODEL_FROM)


# getting input from terminal
args = helpers.argument_parser()
path = args.inp_dir

# checking image is valid or not
if helpers.check_input(path):
    im = helpers.convert_image(path)
    # print the prediction
    predic = loaded_model.predict(im)
    print(CLASSES[np.array(predic).argmax()])

else:
    print ("FAILED TO LOAD THE IMAGE")

