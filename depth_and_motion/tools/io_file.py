#
# Author: Denis Tananaev
# Date: 29.03.2020
#
import numpy as np
import tensorflow as tf
import cv2
import os
import json

def load_image_tf(image_filename, shape, channels=3, ext="png", method="bilinear"):
    """
    shape: img_height, img_width
    """
    img_file = tf.io.read_file(image_filename)
    if ext == "jpeg":
        image = tf.io.decode_jpeg(img_file, channels=channels)
    elif ext == "png":
        image = tf.io.decode_png(img_file, channels=channels)
    else:
        raise ValueError(f"Unknown extention {ext}")
    image =  tf.image.resize(image, shape, method=method)
    image = tf.cast(image, tf.float32)
    return image

def load_dataset_list(dataset_dir, dataset_file, delimiter=";"):
    """
    The function loads list of data from dataset
    file.
    Args:
     dataset_file: path to the .dataset file.
    Returns:
     dataset_list: list of data.
    """

    def add_path_prefix(item):
        """
        Add full path to the data entry
        """
        return os.path.join(dataset_dir, item)

    file_path = os.path.join(dataset_dir, dataset_file)
    dataset_list = []
    with open(file_path) as f:
        dataset_list = f.readlines()
    dataset_list = [x.strip().split(delimiter) for x in dataset_list]
    dataset_list = [list(map(add_path_prefix, x)) for x in dataset_list]

    return dataset_list


def save_dataset_list(dataset_file, data_list):
    """
    Saves dataset list to file.
    """
    with open(dataset_file, "w") as f:
        for item in data_list:
            f.write("%s\n" % item)


def save_depth_16bit(depth_filename, depth):
    depth = np.array(depth, dtype=np.uint16)
    cv2.imwrite(depth_filename, depth)


def save_to_json(json_filename, dict_to_save):
    """
    Save to json file
    """
    with open(json_filename, "w") as f:
        json.dump(dict_to_save, f, indent=2)


def load_from_json(json_filename):
    """
    load from json file
    """
    with open(json_filename) as f:
        data = json.load(f)
        return data