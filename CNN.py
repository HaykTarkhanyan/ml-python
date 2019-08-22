import pandas as pd
import numpy as np
import cv2
import keras
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D

from keras.preprocessing.image import img_to_array

from keras.models import model_from_json

# import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train = data["label"]
del data['label']

num_classes = 10
x_train = data.to_numpy()
# convert labels to vector of zeros and one
y_train = keras.utils.to_categorical(y_train, num_classes)

visualizing an example
plt.imshow(x_train[0], cmap="Greys")
plt.show()
plt.imshow(x_train[2], cmap="Greys")
plt.show()


ConvNet is taken from keras documentation

input_shape = x_train[0].shape

model = Sequential()

x_train = np.array([i.reshape((28, 28, 1)) for i in x_train])

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

visualizing loss and accuract
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
json_file = open('model_cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_cnn.h5")
print("Loaded model from disk")


parser = argparse.ArgumentParser()

parser.add_argument('-inp_dir', type=str, required=True,
                    help="specify path to image")

args = parser.parse_args()
path = args.inp_dir

# visualaize
# plt.imshow(test[ex].reshape((28,28)))
# plt.show()

im = cv2.imread(path)
im.resize(28, 28, 1)
im = img_to_array(im)
im = np.array([im])


print(loaded_model.predict(im).argmax())
