from datetime import datetime

import keras.applications.vgg16
from DataUtils import DataPreprocessing
from ImageUtils import ImageUtils
from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense
from keras.optimizers import SGD

img_width, img_height = 224, 224
num_channels = 3
train_data = '../data/train'
valid_data = '../data/valid'
num_classes = 196
num_train_samples = 6549
num_valid_samples = 1595
batch_size = 16
num_epochs = 100
patience = 50
number_of_classes = 196
number_of_train = 200
if __name__ == '__main__':
    print("reading image paths===================")
    print(keras.backend.backend())
    train_image_paths = ImageUtils.get_image_paths('../data/train', do_shuffle=True)
    test_image_paths = ImageUtils.get_image_paths('../data/valid', do_shuffle=True)

    train_image_paths = train_image_paths[:number_of_train]
    print("reading training set===================")
    x_train = ImageUtils.read_multi_image(train_image_paths)
    y_train = DataPreprocessing.get_one_vs_hot_labels(train_image_paths)

    # print("reading test set===================")
    # x_test = ImageUtils.read_multi_image(test_image_paths)
    # y_test = DataPreprocessing.get_one_vs_hot_labels(test_image_paths)
    # # build up model
    # model = Sequential()
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(25088, activation='relu'))
    model = Sequential()
    conv_layers = keras.applications.VGG16(include_top=False)
    #add convolution layers
    model.add(conv_layers)

    #add fully connected layers
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='Logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'Logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'Models/vgg_with_augmented_data'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.h5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    print("training===================")
    # fine tune the model

    model.fit(
        x_train, y_train, epochs=num_epochs,
        callbacks=callbacks)

    time_stamp = datetime.now().time()  # time object
    model.save('Models/' + 'final_vgg_model_transfer_learning.h5' + str(time_stamp))
