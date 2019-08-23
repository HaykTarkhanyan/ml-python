import os
import cv2
import keras
import argparse
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, Dropout, MaxPooling2D

# Model will be saved in ckpt folder
FOLDER_TO_SAVE = 'ckpt'

# load data from csv
data = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))

# get image labels
y_train = data["label"]
del data['label']

num_classes = 10
x_train = data.to_numpy()
x_train = np.array([i.reshape((28, 28, 1)) for i in x_train])

# convert labels to vector of zeros and one
y_train = keras.utils.to_categorical(y_train, num_classes)

# ConvNet is taken from keras documentation
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 ))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='AdaDelta',
              loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=80,
                    )


# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(FOLDER_TO_SAVE, "model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join(FOLDER_TO_SAVE, "model.h5"))
print("Saved model to ckpt folder")
