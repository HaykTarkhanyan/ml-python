import keras
import argparse
import numpy as np
from utils import helpers
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


args = helpers.argument_parser_for_train()

PATH_TO_SAVE = args.model_save_dir
PATH_TO_LOAD_DATA = args.data_dir

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_CLASSES = 10

# importing data
data_get = ImageDataGenerator()

datagen = ImageDataGenerator(validation_split=0.2)

train_generator = datagen.flow_from_directory(
    PATH_TO_LOAD_DATA,
    target_size=(25, 25),
    batch_size=BATCH_SIZE,
)

valid = datagen.flow_from_directory(
    PATH_TO_LOAD_DATA,
    target_size=(25, 25),
)


# Network that does all the job
classifier = helpers.keras_model(NUM_CLASSES)

classifier.compile(
    optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy']
    )

history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=5000 // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid,
    validation_steps=1000 // BATCH_SIZE,
)


# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(PATH_TO_SAVE), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join(PATH_TO_SAVE))
print("Saved model ")

