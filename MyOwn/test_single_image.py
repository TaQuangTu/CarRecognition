import numpy as np
import keras.models
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array

from DataUtils.DataPreprocessing import get_class_names

model_path = "Models/ResNet.19-0.86.hdf5"

model = keras.models.load_model(model_path)

test_image_path = "../data/test/00001.jpg" # pass your image path here
image = load_img(test_image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, 0)
image = imagenet_utils.preprocess_input(image)

predict = model.predict(image)
max_index = np.argmax(predict)

print(predict)
print(max_index)
class_names= get_class_names('../devkit/cars_meta.mat')
print(class_names[max_index])