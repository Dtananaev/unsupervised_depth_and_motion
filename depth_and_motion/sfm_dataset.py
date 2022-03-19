#
# Author: Denis Tananaev
# Date: 18.02.2021
#

import tensorflow as tf
import numpy as np
from PIL import Image
from depth_and_motion.tools.io_file import load_dataset_list
from depth_and_motion.parameters import Parameters
from depth_and_motion.tools.channels_tools import to_channels_first_np
from tqdm import tqdm
import argparse


class SfmDataset:
    """
    This is dataset layer for sfm experiment
    Arguments:
        param_settings: parameters of experiment
        augmentation: apply augmentation True/False
        shuffle: shuffle the data True/False
    """

    def __init__(self, dataset_dir, dataset_file, batch_size, data_format, input_shape, shuffle=False, random_seed=None):
        # Private methods
        self.in_width, self.in_height = input_shape
        self.dataset_dir = dataset_dir
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.data_format = data_format
        self.random_seed = random_seed

        self.inputs_list = load_dataset_list(self.dataset_dir, dataset_file)
        self.num_samples = len(self.inputs_list)
        self.num_it_per_epoch = int(self.num_samples / self.batch_size)

        self.output_types = [tf.float32, tf.float32, tf.int32]

        ds = tf.data.Dataset.from_tensor_slices(self.inputs_list)

        if shuffle:
            ds = ds.shuffle(self.num_samples)
        ds = ds.map(
            map_func=lambda x: tf.py_function(
                self.load_data, [x], Tout=self.output_types
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.dataset = ds

    def load_data(self, data_input):
        """
        Loads image resizes it and normalizes between 0 and 1
        Note: This is numpy function.
        """
        data = np.asarray(data_input).astype("U")
        data_list = []

        for i in range(len(data)-1):
            images = self.read_and_resize_image(data[i])
            data_list.append(images)
        data_list.append(data[-1])
        return data_list
    
    def read_and_resize_image(self, image_filename):
            img = Image.open(image_filename)
            images = np.asarray(img.resize((self.in_width, self.in_height),Image.BILINEAR), dtype=np.float32) / 255.0
            if self.data_format== "channels_first":
                images = to_channels_first_np(images)
            return images
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetLayer.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="creates .dataset file",
        default="train.datalist",
    )
    args = parser.parse_args()

    param = Parameters().settings
    input_shape = (param["data_width"], param["data_height"])
    train_dataset = SfmDataset(param["dataset_dir"], args.dataset_file, param["batch_size"], param["data_format"], input_shape)

    for samples in tqdm(train_dataset.dataset, total=train_dataset.num_it_per_epoch):
        img_first, img_second, idx = samples
        print(f"first {img_first.shape}, second {img_second.shape} idx {idx}")
        input()