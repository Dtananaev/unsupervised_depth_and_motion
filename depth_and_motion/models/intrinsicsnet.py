#
# Author: Denis Tananaev
# Date: 14.03.2021
#

import tensorflow as tf
from tensorflow.keras import Model

class IntrinsicsNet(Model):
    def __init__(self, name, max_video_index, data_format):
        """
        This is entrinsics table model. It ctreates a table of learnable variables for number of different youtube videos
        we assume each video has its own intrinsics.
        Arguments:
            Name: name of the table
            max_video_index: number of videos (or dataset) 
        """
        super(IntrinsicsNet, self).__init__(name=name)
        self.max_video_index = max_video_index
        self.data_format = data_format
        self.intrinsics_factors, self.distortion = self._create_intrinsics(max_video_index)


    def _create_intrinsics(self, max_video_index):
        intrin_initializer = tf.tile([[1.0, 1.0, 0.5, 0.5]], [max_video_index, 1])
        intrin_factors = tf.Variable(initial_value=intrin_initializer, name="intrinsics_mat")
        dist_initializer = tf.zeros(max_video_index)
        distortion = tf.Variable(initial_value=dist_initializer, name="distrotion")
        return intrin_factors, distortion

    def _get_intrinsics_from_coefficients(self, coefficients, height, width):
        fx_factor, fy_factor, x0_factor, y0_factor = tf.unstack(coefficients, axis=1)
        fx = fx_factor * 0.5 * (height + width)
        fy = fy_factor * 0.5 * (height + width)
        x0 = x0_factor * width
        y0 = y0_factor * height
        return fx, fy, x0, y0


    def make_intrinsics_mat(self, resolution, video_index):
        intrinsics_factors = tf.gather(self.intrinsics_factors, video_index, axis=0)
        fx, fy, x0, y0 = self._get_intrinsics_from_coefficients(intrinsics_factors, resolution[0], resolution[1])
        zero = tf.zeros_like(fx)
        one = tf.ones_like(fx)
        int_mat = [[fx, zero, x0], [zero, fy, y0], [zero, zero, one]]
        int_mat = tf.transpose(int_mat, [2, 0, 1])
        return int_mat


    def get_resolution(self, input_images):
        """
        The function returns the img height img width of the input image
        """
        shape = tf.shape(input_images)
        spatial_shape = tf.cond(tf.math.equal(self.data_format, "channels_first"), lambda: [shape[2], shape[3]], lambda: [shape[1], shape[2]])
        resolution = tf.cast(spatial_shape, tf.float32)
        return resolution


    def call(self, x, training):
        input, video_idx = x
        resolution = self.get_resolution(input)
        intr_mat = self.make_intrinsics_mat(resolution, video_idx)
        distortion = tf.gather( self.distortion, video_idx)
        distortion = tf.expand_dims(tf.expand_dims(distortion, axis=-1), axis=-1)
        return intr_mat, distortion




