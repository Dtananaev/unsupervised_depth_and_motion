#
# Author: Denis Tananaev
# Date: 19.02.2021
#

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D 
from tensorflow.keras.regularizers import l2
from dl_app_backend.tools.tensors_tools import resize_tensor


class MotionNet(Model):
    """
    This is implementation of the motion field net from the paper:
    https://arxiv.org/abs/2010.16404
    Arguments:
        data_format: format of the input tensors (can be channels_first or channels_last)
        weight_decay: l2 weight decay
        name: the name of the network
        constraint_minimum: the lower bound value which constraint
        the learned scale values for rotation and translation
        auto_mask: True to automatically masking out the residual translations
        by thresholding on their mean values.
    Returns:
        rotation: the camera rotation Euler angles with format [B, 1, 1, 3] or [B, 3 ,1, 1]
        translation: the camera translation with format [B, 1, 1, 3] or [B, 3 ,1, 1]
        residual_translation: the residual translation filed for moving objects shape [B, H, W, 3] or [B, 3, H, W]
        intrinsics_mat: the intrinsics matrix [B, 3, 3]
    """
    def __init__(self, data_format, weight_decay, name, constraint_minimum=0.001, auto_mask=False):
        super(MotionNet, self).__init__(name=name)
        self.rot_scale, self.trans_scale = self._create_scales(constraint_minimum)
        self.encoder = MotionNetEncoder(data_format=data_format, weight_decay=weight_decay,  name="MotionNetEncoder")
        self.pose_head = PoseHead(data_format=data_format, name="PoseHead")
        #self.intrinsics_head = IntrinsicsHead(data_format=data_format, name="IntrinsicsHead")
        self.residual_translation_head = ResidualTranslationHead(data_format=data_format, weight_decay=weight_decay, auto_mask=auto_mask,  name="ResidualTranslationHead")
    
    
    def _create_scales(self, constraint_minimum):
        """
        Creates constraint learnable variables representing rotation and translation scaling factors.
        Arguments:
            constraint_minimum: the constraint which not allow learnable variable be lower than this value
        """
        def constraint(x):
            return tf.nn.relu(x - constraint_minimum) + constraint_minimum
        rot_scale = tf.Variable(initial_value=0.01, name='Rotation', constraint=constraint)
        trans_scale = tf.Variable(initial_value=0.01,  name='Translation', constraint=constraint)
        return rot_scale, trans_scale

    
    def call(self, x):
        input = x
        bottleneck, encoder_features = self.encoder(x)
        # 1. Camera pose
        rotation, translation = self.pose_head(bottleneck)
        # 2. Camera intrinsics matrix
        #intrinsics_mat = self.intrinsics_head((bottleneck, input))
        # 3. Get residual tranlslation of the moving objects
        residual_translation = self.residual_translation_head((translation, input, encoder_features))
        # Scale up
        rotation *= self.rot_scale
        translation *= self.trans_scale
        residual_translation *= self.trans_scale

        return rotation, translation, residual_translation

class MotionNetEncoder(Layer):
    """
    The motion network encoder part
    Arguments:
        data_format: format of the input tensors (can be channels_first or channels_last)
        weight_decay: l2 weight decay
        name: the name of the layer
        filters: the output number of filters for each layer
        activation: activation function.
    """
    def __init__(self, data_format, weight_decay,  name,  filters=[16, 32, 64, 128, 256, 256, 256], activation=tf.nn.relu):
        super(MotionNetEncoder, self).__init__(name=name)
        # Shortcut for l2 weigt decay
        l2_wd = l2(weight_decay)
        if data_format == "channels_first":
            self.spatial_axis = [2, 3]
        else:
            self.spatial_axis = [1, 2]
        # The network layers
        self.conv_1 = Conv2D(filters[0], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_1")
        self.conv_2 = Conv2D(filters[1], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_2")
        self.conv_3 = Conv2D(filters[2], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_3")
        self.conv_4 = Conv2D(filters[3], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_4")
        self.conv_5 = Conv2D(filters[4], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_5")
        self.conv_6 = Conv2D(filters[5], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_6")
        self.conv_7 = Conv2D(filters[6], 3, 2, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_7")

    def call(self, x, training):
        x = conv1 = self.conv_1(x)
        x = conv2 = self.conv_2(x)
        x = conv3 = self.conv_3(x)
        x = conv4 = self.conv_4(x)
        x = conv5 = self.conv_5(x)
        x = conv6 = self.conv_6(x)
        conv7 = self.conv_7(x)
        bottleneck = tf.reduce_mean(conv7, axis=self.spatial_axis, keepdims=True)
        return bottleneck, (conv7, conv6, conv5, conv4, conv3, conv2, conv1)


class PoseHead(Layer):
    def __init__(self,data_format,  name):
        super(PoseHead, self).__init__(name=name)
        if data_format == "channels_first":
            self.axis = 1
            self.spatial_axis=(2, 3)
        else:
            self.axis = -1 
            self.spatial_axis=(1, 2)

        self.background_motion =  Conv2D(6, 1, 1, kernel_regularizer=None, data_format=data_format, padding="same", activation=None, name="background_motion")
    def call(self, x):
        background_motion = self.background_motion(x)
        rotation, translation = tf.split(background_motion, 2, axis=self.axis)
        rotation = tf.squeeze(rotation, axis= self.spatial_axis)
        return rotation, translation
    

class IntrinsicsHead(Layer):
    """
    This is intrinsics head layer.
    Arguments:
        data_format: the data format.
        name: the name of the network.
    Returns:
        intrinsic_mat: Tensor of shape [B, 3, 3], and type float32,
        where the 3x3 part is (fx, 0, cx), (0, fy, cy), (0, 0, 1).
    """
    def __init__(self, data_format,  name):
        super(IntrinsicsHead, self).__init__(name=name)
        self.data_format = data_format
        if data_format == "channels_first":
            self.spatial_axis = (2, 3)
        else:
            self.spatial_axis = (1, 2)

        self.conv_foc =  Conv2D(2, 1, 1, kernel_regularizer=None, data_format=data_format, padding="same", activation=tf.nn.softplus, name="foc")
        self.offsets = Conv2D(2, 1, 1, kernel_regularizer=None, data_format=data_format, padding="same", activation=None, name="offsets")

    
    def get_resolution(self, input_images):
        """
        The function returns the img height img width of the input image
        """
        shape = tf.shape(input_images)
        spatial_shape = tf.cond(tf.math.equal(self.data_format, "channels_first"), lambda: [shape[2], shape[3]], lambda: [shape[1], shape[2]])
        resolution = tf.cast([spatial_shape], tf.float32)
        return resolution

    def call(self, x):
        bottleneck, input_images = x
        resolution = self.get_resolution(input_images)

        # Since the focal lengths in pixels tend to be in the order of magnitude of
        # the image width and height, we multiply the network prediction by them.
        before = self.conv_foc(bottleneck)

        focal_lengths = tf.squeeze(before, axis=self.spatial_axis) * resolution
        # The pixel offsets tend to be around the center of the image, and they
        # are typically a fraction the image width and height in pixels. We thus
        # multiply the network prediction by the width and height, and the
        # additional 0.5 them by default at the center of the image.
        offsets = tf.squeeze(self.offsets(bottleneck) + 0.5, axis=self.spatial_axis) * resolution
        foci = tf.linalg.diag(focal_lengths)
        # Create intrinsics matrix
        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
        batch_size = tf.shape(bottleneck)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat


class ResidualTranslationHead(Layer):
    """
    The residual translation head layer
    Arguments:
        data_format: the data format.
        weight_decay: l2 weight decay.
        name: the name of the network.
        auto_mask: True to automatically masking out the residual translations
        by thresholding on their mean values.
        filters: sizes of the output filters.
    Returns:
      residual_translation: the translation of the moving object [B, H, W, 3] or [B,3, H, W]      
    """
    def __init__(self, data_format, weight_decay, name, auto_mask=False, filters=[16, 32, 64, 128, 256, 256, 256]): #[16, 32, 64, 128, 256, 512, 1024]
        super(ResidualTranslationHead, self).__init__(name=name)
        self.auto_mask = auto_mask
        if data_format == "channels_first":
            self.axis = 1
            self.spatial_axis = [2, 3]
        else:
            self.axis = -1 
            self.spatial_axis = [1, 2]
        # The network layers
        self.residual_translation = Conv2D(3, 1, 1, kernel_regularizer=None, data_format=data_format, padding="same", activation=None, name="unrefined_residual_translation")
        self.refine_7 = RefineMotionField(weight_decay, filters[6], data_format=data_format, name="refine_7")
        self.refine_6 = RefineMotionField(weight_decay, filters[5], data_format=data_format, name="refine_6")
        self.refine_5 = RefineMotionField(weight_decay, filters[4], data_format=data_format, name="refine_5")
        self.refine_4 = RefineMotionField(weight_decay, filters[3], data_format=data_format, name="refine_4")
        self.refine_3 = RefineMotionField(weight_decay, filters[2], data_format=data_format, name="refine_3")
        self.refine_2 = RefineMotionField(weight_decay, filters[1], data_format=data_format, name="refine_2")
        self.refine_1 = RefineMotionField(weight_decay, filters[0], data_format=data_format, name="refine_1")
        self.refine_images = RefineMotionField(weight_decay, 4, data_format=data_format, name="refine_images")

    
    def auto_mask_residual_translation(self, residual_translation):
        sq_residual_translation = tf.sqrt(
            tf.reduce_sum(residual_translation ** 2, axis=self.axis, keepdims=True))
        mean_sq_residual_translation = tf.reduce_mean(sq_residual_translation, axis=self.spatial_axis, keepdims=True)
        # A mask of shape [B, h, w, 1]
        mask_residual_translation = tf.cast(sq_residual_translation > mean_sq_residual_translation, residual_translation.dtype.base_dtype)
        residual_translation *= mask_residual_translation
        return residual_translation


    def call(self, x):
        translation, input_images, encoder_features = x
        conv7, conv6, conv5, conv4, conv3, conv2, conv1 = encoder_features
        # Get residual translation
        residual_translation = self.residual_translation(translation)
        residual_translation = self.refine_7((residual_translation, conv7))
        residual_translation = self.refine_6((residual_translation, conv6))
        residual_translation = self.refine_5((residual_translation, conv5))
        residual_translation = self.refine_4((residual_translation, conv4))
        residual_translation = self.refine_3((residual_translation, conv3))
        residual_translation = self.refine_2((residual_translation, conv2))
        residual_translation = self.refine_1((residual_translation, conv1))
        residual_translation = self.refine_images((residual_translation, input_images))
        if self.auto_mask:
            residual_translation = self.auto_mask_residual_translation(residual_translation)
        return residual_translation


class RefineMotionField(Layer):
    """Refines a motion field using features from another layer.

        This function builds an element of a UNet-like architecture. `motion_field`
        has a lower spatial resolution than `layer`. First motion_field is resized to
        `layer`'s spatial resolution using bilinear interpolation, then convolutional
        filters are applied on `layer` and the result is added to the upscaled
        `motion_field`.

        This scheme is inspired by FlowNet (https://arxiv.org/abs/1504.06852), and the
        realization that keeping the bottenecks at the same (low) dimension as the
        motion field will pressure the network to gradually transfer details from
        depth channels to space.

        The specifics are slightly different form FlowNet: We use two parallel towers,
        a 3x3 convolution, and two successive 3x3 convolutions, as opposed to one
        3x3 convolution in FLowNet. Also, we add the result to the upscaled
        `motion_field`, forming a residual connection, unlike FlowNet. These changes
        seemed to improve the depth prediction metrics, but exploration was far from
        exhaustive.
    """
    def __init__(self, weight_decay, out_channels, name, data_format, activation=None):
        super(RefineMotionField, self).__init__(name=name)
        self.data_format = data_format
        if data_format == "channels_first":
            self.axis = 1
        else:
            self.axis = -1 
        l2_wd = l2(weight_decay)
        self.conv_t1 = Conv2D(out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_t1")
        self.conv_t2_1 = Conv2D(out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_t2_1")
        self.conv_t2_2 = Conv2D(out_channels, 3, 1, kernel_regularizer=l2_wd, data_format=data_format, padding="same", activation=activation, name="conv_t2_2")
        self.conv_res = Conv2D(3, 1, 1, kernel_regularizer=None, data_format=data_format, padding="same", activation=None, name="conv_res")


    def call(self, x):
        motion_field, features = x
        shape = tf.shape(features)
        spatial_shape = tf.cond(tf.math.equal(self.data_format, "channels_first"),lambda: [shape[2], shape[3]], lambda:   [shape[1], shape[2]])

        upscaled_motion_field = resize_tensor(motion_field, spatial_shape, data_format=self.data_format, method="bilinear")
        conv_input = tf.concat([upscaled_motion_field, features], axis=self.axis)
        t1 =  self.conv_t1(conv_input)
        t2 = self.conv_t2_1(conv_input)
        t2 = self.conv_t2_2(t2)
        conv_output = tf.concat([t1, t2], axis=self.axis)
        conv_output = self.conv_res(conv_output)
        out  = upscaled_motion_field + conv_output
        return out
