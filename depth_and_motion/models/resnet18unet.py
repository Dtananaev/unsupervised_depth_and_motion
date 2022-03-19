#
# Author: Denis Tananaev
# Date: 23.02.2021
#

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer, UpSampling2D, BatchNormalization, MaxPool2D, Conv2DTranspose, ReLU, LayerNormalization
from tensorflow.keras.regularizers import l2

class RandomizedLayerNormalization(Layer):
    def __init__(self, name, axis, max_stdev=0.5, variance_epsilon=1e-3, rampup_steps=10000.0):
        """
        Applies layer normalization and applies noise on the mean and variance.
        For every item in a batch and for every layer, we calculate the mean and
        variance across the spatial dimensions, and multiply them by Gaussian noise
        with a mean equal to 1.0 (at training time only). This improved the results
        compared to batch normalization - see more in
        https://arxiv.org/abs/1904.04998.
        Arguments:
        name: the name of the layers
        axis: axis 1 or -1 corresponds to channels dimension
        max_stdev: maximal stdev
        variance_epsilon: variance_epsilon
        rampup_steps: number of steps to increase noise from 0 to max_stdev

        """
        super(RandomizedLayerNormalization, self).__init__(name=name)
        self.axis = axis
        if axis == 1:
            self.spatial_axis = [2, 3]
        else:
            self.spatial_axis = [1, 2]

        self.max_stdev = max_stdev
        self.variance_epsilon = variance_epsilon
        self.call_counter = 0.0
        self.rampup_steps = rampup_steps

    def build(self, input_shape):
        params_shape = input_shape[self.axis]
        self.beta = self.add_weight(name='beta',shape=params_shape, initializer=tf.initializers.zeros())
        self.gamma = self.add_weight(name='gamma',shape=params_shape, initializer=tf.initializers.ones())

    def get_stdev(self):
        self.call_counter += 1.0 
        return self.max_stdev * tf.math.square(tf.math.minimum(self.call_counter / self.rampup_steps, 1.0))

    def call(self, x, training):
        mean, variance = tf.nn.moments(x, self.spatial_axis, keepdims=True)
        if training:
            stdev = self.get_stdev()
            mean *= 1.0 + tf.random.truncated_normal(tf.shape(mean), stddev=stdev)
            variance *= 1.0 + tf.random.truncated_normal(tf.shape(variance), stddev=stdev)
        outputs = tf.nn.batch_normalization(x, mean, variance, offset=self.beta, scale=self.gamma, variance_epsilon=self.variance_epsilon)
        return outputs

class ResNet18Unet(Model):
    """
        A depth prediciton network based on a ResNet18 UNet architecture.

        This network is identical to disp_net in struct2depth.nets with
        architecture='resnet', with the following differences:

        1. We use a softplus activation to generate positive depths. This eliminates
            the need for the hyperparameters DISP_SCALING and MIN_DISP defined in
            struct2depth.nets. The predicted depth is no longer bounded.

        2. The network predicts depth rather than disparity, and at a single scale.
    """
    def __init__(self,data_format, weight_decay, name):
        super(ResNet18Unet, self).__init__(name=name)
        self.encoder= ResNet18Encoder(data_format=data_format, weight_decay=weight_decay, name="ResNet18Encoder")
        self.decoder= UnetDecoder(data_format=data_format, weight_decay=weight_decay,  name="UnetDecoder")


    def call(self, x, training=False):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x 


class ResNet18Encoder(Layer):
    """
    Defines a ResNet18-based encoding architecture.

    This implementation follows Juyong Kim's implementation of ResNet18 on GitHub:
    https://github.com/dalgu90/resnet-18-tensorflow

    """

    def __init__(
        self,
        data_format,
        weight_decay,
        name,
        filters=[64, 64, 128, 256, 512],
        activation_fn=ReLU,
        normalization_fn=BatchNormalization,
    ):
        super(ResNet18Encoder, self).__init__(name=name)
        if data_format == "channels_first":
            self.axis = 1
        else:
            self.axis = -1 
        l2_wd = l2(weight_decay)
        # Conv 1
        self.conv_1 = Conv2D(
            filters[0], 7, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_1")
        self.bn_1 = normalization_fn(name="bn_1", axis=self.axis)
        self.relu_1 = activation_fn(name="relu_1")
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=data_format)
        # Conv 2
        self.res2_1 = ResidualBlock(weight_decay, filters[1], name="res2_1", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        self.res2_2 = ResidualBlock(weight_decay, filters[1], name="res2_2", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        # Conv 3
        self.res3_1 = ResidualBlockFirst(weight_decay, filters[2], 2, name="res3_1", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        self.res3_2 = ResidualBlock(weight_decay,  filters[2], name="res3_2", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        # Conv 4
        self.res4_1 = ResidualBlockFirst(weight_decay, filters[3], 2,name="res4_1", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        self.res4_2 = ResidualBlock(weight_decay,  filters[3], name="res4_2", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        # Conv 5
        self.res5_1 = ResidualBlockFirst(weight_decay, filters[4], 2, name="res5_1", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)
        self.res5_2 = ResidualBlock(weight_decay,  filters[4], name="res5_2", normalization_fn=normalization_fn, activation_fn=activation_fn, data_format=data_format)

    def call(self, x, training=False):
        # Conv 1
        x = self.conv_1(x)
        x = self.bn_1(x,training=training)
        x = econv1 = self.relu_1(x)
        x = self.maxpool(x)
        # Conv 2
        x = self.res2_1(x,training=training)
        x = econv2 = self.res2_2(x,training=training)
        # Conv 3
        x = self.res3_1(x,training=training)
        x = econv3 = self.res3_2(x,training=training)
        # Conv 4
        x = self.res4_1(x,training=training)
        x = econv4 = self.res4_2(x,training=training)
        # Conv 5
        x = self.res5_1(x, training=training)
        econv5 = self.res5_2(x,training=training)
        return econv5, (econv4, econv3, econv2, econv1)


class UnetDecoder(Layer):
    """A depth prediciton network based on a ResNet18 UNet architecture.
    1. We use a softplus activation to generate positive depths. This eliminates
        the need for the hyperparameters DISP_SCALING and MIN_DISP defined in
        struct2depth.nets. The predicted depth is no longer bounded.
    2. The network predicts depth rather than disparity, and at a single scale.
    Args:
    weight_decay: weight decay
    padding_mode: A boolean, if True, deconvolutions will be padded in
        'REFLECT' mode, otherwise in 'CONSTANT' mode (the former is not supported
        on  TPU)

    Returns:
    A tf.Tensor of shape [B, H, W, 1] containing depths maps.
    """
    def __init__(
        self,
        data_format,
        weight_decay,
        name,
        filters=[16, 32, 64, 128, 256],
        activation_fn=None,
        padding_mode="CONSTANT",

    ):
        super(UnetDecoder, self).__init__(name=name)
        self.data_format = data_format
        self.mode = padding_mode
        l2_wd = l2(weight_decay)
        if data_format == "channels_first":
            self.axis = 1
        else:
            self.axis = -1 
        self.upconv_5 = Conv2DTranspose(filters[4], 3, 2, padding="same", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format, name="upconv_5")
        self.iconv_5 = Conv2D(filters[4], 3, 1, padding="valid", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format,  name="iconv_5")

        self.upconv_4 = Conv2DTranspose(filters[3], 3, 2, padding="same", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format, name="upconv_4")
        self.iconv_4 = Conv2D(filters[3], 3, 1, padding="valid", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format,  name="iconv_4")

        self.upconv_3 = Conv2DTranspose(filters[2], 3, 2, padding="same", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format, name="upconv_3")
        self.iconv_3 = Conv2D(filters[2], 3, 1, padding="valid", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format,  name="iconv_3")

        self.upconv_2 = Conv2DTranspose(filters[1], 3, 2, padding="same", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format, name="upconv_2")
        self.iconv_2 = Conv2D(filters[1], 3, 1, padding="valid", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format,  name="iconv_2")

        self.upconv_1 = Conv2DTranspose(filters[0], 3, 2, padding="same", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format, name="upconv_1")
        self.iconv_1 = Conv2D(filters[0], 3, 1, padding="valid", kernel_regularizer=l2_wd, activation=activation_fn, data_format=data_format,  name="iconv_1")

        self.depth_output = Conv2D(1, 3, 1, padding="valid", activation=tf.nn.softplus, data_format=data_format,  name="depth_output")

    def _pad(self, x, mode):
        if self.data_format == "channels_first":
            pad = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], mode=mode)
        else:
            pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=mode)
        return pad

    def call(self, input, training=False):
        bottleneck, skip_connections = input
        econv4, econv3, econv2, econv1 = skip_connections

        x = self.upconv_5(bottleneck)

        x = self._pad(tf.concat([x, econv4], axis=self.axis), self.mode) 
        x = self.iconv_5(x)

        x = self.upconv_4(x)

        x = self._pad(tf.concat([x, econv3], axis=self.axis), self.mode) 
        x = self.iconv_4(x)

        x = self.upconv_3(x)

        x = self._pad(tf.concat([x, econv2], axis=self.axis), self.mode) 
        x = self.iconv_3(x)

        x = self.upconv_2(x)

        x = self._pad(tf.concat([x, econv1], axis=self.axis), self.mode) 
        x = self.iconv_2(x)

        x = self.upconv_1(x)
        x = self._pad(x, self.mode)
        x = self.iconv_1(x)

        x = self._pad(x, self.mode)
        x = self.depth_output(x)
        return x


class ResidualBlockFirst(Layer):
    """
    The resnet18 residual block first.
    
    Arguments:
        weight_decay: scalar for weight decay l2 regularization
        out_channels: num of channels in output tensor
        strides: scalar number for stride
        name: name of the layer
        activation: activation function
        normalization_fn: normalization function of the network
        data_format: format of the data tensor
    """

    def __init__(
        self,
        weight_decay,
        out_channels,
        strides,
        name,
        activation_fn=ReLU,
        normalization_fn=BatchNormalization,
        data_format="channels_last",
    ):
        super(ResidualBlockFirst, self).__init__(name=name)
        if data_format == "channels_first":
            self.axis = 1
        else:
            self.axis = -1 
        self.out_channels = out_channels
        self.strides = strides
        # l2 weigt decay
        l2_wd = l2(weight_decay)

        # The network layers
        self.conv_shortcut = Conv2D(
            out_channels, 1, strides, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_shortcut")
        self.maxpool = MaxPool2D(strides=(strides, strides), padding='valid', data_format=data_format)
        self.conv_1 = Conv2D(
            out_channels, 3, strides, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_1")
        self.bn_1 = normalization_fn(name="bn_1", axis=self.axis)
        self.relu_1 = activation_fn(name="relu_1")
        self.conv_2 = Conv2D(
            out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_2")
        self.bn_2 = normalization_fn(name="bn_2", axis=self.axis)
        self.relu_2 = activation_fn(name="relu_2")

    def _get_shortcut(self, x):
        """
        The get shortcut function
        if in_channels == self.out_channels:
            if self.strides == 1:
               shortcut = tf.identity(x)
            else:
                shortcut = self.maxpool(x)
            else:
                shortcut = self.conv_shortcut(x)
        Tf autograph implementation below.
        """
        in_channels = tf.shape(x)[self.axis]
        shortcut = tf.cond(tf.math.equal(self.strides, 1), lambda: tf.identity(x), lambda: self.maxpool(x))
        return tf.cond(tf.math.equal(in_channels, self.out_channels), lambda:  shortcut, lambda: self.conv_shortcut(x))

    def call(self, x, training=False):
        shortcut = self._get_shortcut(x)
        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = x + shortcut
        x = self.relu_2(x)
        return x


class ResidualBlock(Layer):
    """
    The resnet18 residual block.
    
    Arguments:
        weight_decay: scalar for weight decay l2 regularization
        out_channels: num of channels in output tensor
        strides: scalar number for stride
        name: name of the layer
        activation: activation function
        normalization_fn: normalization function of the network
        data_format: format of the data tensor (note: channels_last is not supported)
    """

    def __init__(
        self,
        weight_decay,
        out_channels,
        name,
        activation_fn=ReLU,
        normalization_fn=BatchNormalization,
        data_format="channels_last",
    ):
        super(ResidualBlock, self).__init__(name=name)
        if data_format == "channels_first":
            self.axis = 1
        else:
            self.axis = -1 
        l2_wd = l2(weight_decay)
        self.conv_1 = Conv2D(
            out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_1")
        self.bn_1 = normalization_fn(name="bn_1", axis=self.axis)
        self.relu_1 = activation_fn(name="relu_1")
        self.conv_2 = Conv2D(
            out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=None, name="conv_2")
        self.bn_2 = normalization_fn(name="bn_2", axis=self.axis)
        self.relu_2 = activation_fn(name="relu_2")

    def call(self, x, training=False):
        shortcut = x
        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = x + shortcut
        x = self.relu_2(x)
        return x


