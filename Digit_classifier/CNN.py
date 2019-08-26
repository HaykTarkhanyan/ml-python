from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D


def keras_model(num_classes):
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

    return model