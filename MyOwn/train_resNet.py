import keras.applications.resnet50
import numpy as np
import os.path
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Flatten, AveragePooling2D, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from numpy.compat import os_PathLike

NUMBER_OF_CLASSES = 196
img_width, img_height = 224, 224
num_channels = 3
train_data = '../data/train'
valid_data = '../data/valid'
num_classes = 196
num_train_samples = 6549
num_valid_samples = 1595
verbose = 1
batch_size = 16
num_epochs = 100
patience = 50
model_path = "Models/Final/ResNet_final.h5"

# build model
model = None
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(model.output)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(NUMBER_OF_CLASSES, activation='softmax', name='fc8')(x_fc)

    model = Model(model.input, x_fc)
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# prepare data for training
train_data_gen = ImageDataGenerator(rotation_range=20.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
valid_data_gen = ImageDataGenerator()
# callbacks
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
model_names = 'Models/ResNet.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [early_stop, reduce_lr]

# generators
train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                     class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                     class_mode='categorical')

# fine tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples / batch_size,
    validation_data=valid_generator,
    validation_steps=num_valid_samples / batch_size,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=verbose)
model.save(model_path)

