import os

import keras
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder


def get_one_vs_hot_labels(image_paths):
    labels = encode_labels_from_paths(image_paths)
    num_of_classes = np.asarray(labels).max()+1  # starting index is from 0
    labels = keras.utils.to_categorical(labels, num_of_classes, dtype=np.int)
    return labels

def encode_labels_from_paths(image_paths):
    labels = [p.split(os.path.sep)[-2] for p in image_paths]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels


def get_class_names(meta_file_path):
    cars_meta = scipy.io.loadmat(meta_file_path)
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    class_names = class_names.reshape(class_names.size)
    return class_names

class_names = get_class_names('../../devkit/cars_meta.mat')
print(class_names)