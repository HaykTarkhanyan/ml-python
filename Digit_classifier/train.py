from utils import helpers

# Model will be saved in ckpt folder by default
FOLDER_TO_SAVE = 'ckpt'
FOLDER_TO_LOAD_DATA = 'data'

# load data
x_train, y_train, test = helpers.load_data(FOLDER_TO_LOAD_DATA)

# ConvNet is taken from keras documentation
num_classes = 10
model = helpers.keras_model(num_classes)

# train the data
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=80,
                    )

# saving the model.json and model.h5
helpers.save_model(FOLDER_TO_SAVE)


