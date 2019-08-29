import keras
import argparse
import numpy as np
from utils import helpers
from ANN import keras_model
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


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_save_dir', type=str, required=False,
                        help="specify location where model will be saved",
                        default='pretrained_model')
    parser.add_argument('-data_dir', type=str, required=False,
                        help="specify path to data",
                        default='data')
    parser.add_argument('-epochs', type=str, required=False,
                        help="specify number of epochs to train",
                        default='7')
    parser.add_argument('-batch_size', type=str, required=False,
                        help="specify batch_size",
                        default='1')

    args = parser.parse_args()
    return args


def load_data(path):    
    # importing data
    data_get = ImageDataGenerator()

    datagen = ImageDataGenerator(validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        path,
        target_size=(25, 25),
        batch_size=BATCH_SIZE,
    )

    valid = datagen.flow_from_directory(
        path,
        target_size=(25, 25),
    )

    return train, valid

def keras_model(train_generator, valid):
    # Network that does all the job
    classifier = keras_model(NUM_CLASSES)

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

    return model, history

def save_model():
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(PATH_TO_SAVE), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(PATH_TO_SAVE))
    print("Saved model ")

if __name__ == "__main__":
    # parse arguments 
    args = argument_parser()
    # load data
    train_generator, valid = load_data(PATH_TO_LOAD_DATA)
    # train model
    model, history = train_model(train_generator, valid)
    # save model
    save_model(model, PATH_TO_SAVE_WEIGHTS, PATH_TO_SAVE_CONFIG)


