
import cv2
from keras_applications import imagenet_utils
from keras_preprocessing.image import load_img, img_to_array
from imutils import paths
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import random

def get_image_paths(directory, do_shuffle=False):
    image_path_list = list(paths.list_images(directory))
    if do_shuffle is True:
        random.shuffle(image_path_list)
    return image_path_list


def read_image(image_path, flags=None):
    if not flags is None:
        image = cv2.imread(image_path, flags)
    else:
        image = cv2.imread(image_path)
    return image


def read_multi_image(image_paths):
    list_images = []
    for (j, imagePath) in enumerate(image_paths):
        print("reading image "+str(j))
        img = load_img(imagePath, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, 0)
        img = imagenet_utils.preprocess_input(img)
        list_images.append(img)
    list_images = np.vstack(list_images)
    return list_images
