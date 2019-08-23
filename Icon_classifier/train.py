import cv2
import keras
import argparse
import numpy as np
from keras import Sequential
from numpy import expand_dims
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import model_from_json
from keras.layers import Conv3D, MaxPooling3D
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# directory for training data
LOAD_DATA_FROM = 'data'

# importing data
data_get = ImageDataGenerator()

datagen = ImageDataGenerator(validation_split=0.2)

train_generator = datagen.flow_from_directory(
    LOAD_DATA_FROM,
    target_size=(25, 25),
    batch_size=8,
)

valid = datagen.flow_from_directory(
    LOAD_DATA_FROM,
    target_size=(25, 25),
)


# Network that does all the job
classifier = Sequential()
batch = 4
num_classes = 5

classifier.add(Dense(100, activation='relu',
                     input_shape=train_generator.image_shape))
classifier.add(Flatten())
classifier.add(Dense(80, activation='relu'))
classifier.add(Dense(80, activation='relu'))
classifier.add(Dense(num_classes,  activation='softmax'))

nadam = keras.optimizers.nadam(lr=.000000753)

classifier.compile(
    optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])

history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=5000 // batch,
    epochs=7,
    validation_data=valid,
    validation_steps=1000 // batch,
)
