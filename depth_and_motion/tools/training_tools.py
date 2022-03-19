#
# Author: Denis Tananaev
# Date: 02.04.2020
#

import tensorflow as tf
import glob
import os


def setup_gpu():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        # Will not allocate all memory but only necessary amount
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def initialize_model(model, width, height, channels, data_format):
    """
    Helper tf2 specific model initialization (need for saving mechanism)
    """
    if data_format == "channels_first":
        input_shape = [1, channels,  height, width]
    else:
        input_shape = [1, height, width, channels]
    sample = tf.zeros(input_shape, tf.float32)
    model.predict(sample)


def load_model(checkpoints_dir, model, resume):
    """
    Resume model from given checkpoint
    """
    start_epoch = 0

    if resume:
        search_string = os.path.join(checkpoints_dir, model.name + "*")
        checkpoints_list = sorted(glob.glob(search_string))
        if len(checkpoints_list) > 0:
            current_epoch = int(os.path.split(checkpoints_list[-1])[-1].split("-")[-1])
            model = tf.keras.models.load_model(checkpoints_list[-1])
            start_epoch = current_epoch + 1  # we should continue from the next epoch
            print(f"RESUME TRAINING FROM CHECKPOINT: {checkpoints_list[-1]}.")
        else:
            print(f"CAN'T RESUME TRAINING! NO CHECKPOINT FOUND! START NEW TRAINING!")
    return start_epoch, model
