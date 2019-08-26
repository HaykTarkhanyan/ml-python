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


# takes data location, path where to save model, 
# num of epochs and batch size
args = helpers.argument_parser_for_train()

FOLDER_TO_SAVE = args.model_save_dir
FOLDER_TO_LOAD_DATA = args.data_dir

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_CLASSES = 10

# load data from csv
data = pd.read_csv(os.path.join(FOLDER_TO_LOAD_DATA))
test = pd.read_csv(os.path.join(FOLDER_TO_LOAD_DATA))

# get image labels
y_train = data["label"]
del data['label']

x_train = data.to_numpy()
x_train = np.array([i.reshape((28, 28, 1)) for i in x_train])
# convert labels to vector of zeros and one
y_train = keras.utils.to_categorical(y_train, num_classes)



model = keras_model(NUM_CLASSES)

model.compile(optimizer='AdaDelta',
              loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    )


# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(FOLDER_TO_SAVE), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join(FOLDER_TO_SAVE))
print("Saved model to ckpt folder")
