from utils import helpers

# directories for data and model future location
LOAD_DATA_FROM = 'data'
FOLDER_TO_SAVE = 'ckpt'

# loading data 
train_generator, valid = helpers.load_data(LOAD_DATA_FROM)

# keras model 
batch = 4
num_classes = 5
classifier = helpers.keras_model(num_classes)

history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=5000 // batch,
    epochs=7,
    validation_data=valid,
    validation_steps=1000 // batch,
)

# saving the model.json and model.h5
helpers.save_model(FOLDER_TO_SAVE)

