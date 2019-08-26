import numpy as np
from utils import helpers

# loading model from specified fodler
LOAD_MODEL_FROM = 'ckpt'
loaded_model = helpers.load_model(LOAD_MODEL_FROM)

# getting input from terminal
args = helpers.argument_parser()
path = args.inp_dir

# before trying to predict making sure that directory is valid
if helpers.check_input(path):
    im = helpers.convert_image(path)
    # print the prediction
    print(loaded_model.predict(im).argmax())
    
else:
    print ("Failed to load image ")