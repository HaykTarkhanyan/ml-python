import keras
from keras.layers import Dense
from keras.layers import Flatten


def keras_model(num_classes):
    # Network that does all the job
    classifier = Sequential()

    classifier.add(Dense(100, activation='relu',
                        input_shape=train_generator.image_shape))
    classifier.add(Flatten())
    classifier.add(Dense(80, activation='relu'))
    classifier.add(Dense(80, activation='relu'))
    classifier.add(Dense(num_classes,  activation='softmax'))

    nadam = keras.optimizers.nadam(lr=.000000753)

    return classifier
