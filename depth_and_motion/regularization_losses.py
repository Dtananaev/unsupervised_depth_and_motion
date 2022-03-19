#
# Author: Denis Tananaev
# Date: 11.03.2021
#
import tensorflow as tf
from depth_and_motion.tools.tensors_tools import get_forward_difference


def edge_aware_smoothness_loss(disp, img, data_format):
    """
    This is edge aware smoothness loss. 
    Arguments:
      disp: disparity of the shape [B, H, W, 1] (or [B, 1, H, W])
      img: input image of the shape [B, H, W, 3] (or [B, 3, H, W])
      data_format: channels_first or channels_last
    Returns:
      smoothness_loss: smoothness_loss penalizes difference in homogeneous areas and not penalize on edges
    """
    if data_format == "channels_first":
        channel_axis = 1
        spatial_axis = [2, 3] 
    else:
        channel_axis = -1
        spatial_axis = [1, 2] 

    mean_disp = tf.reduce_mean(disp, axis=spatial_axis, keepdims=True)
    norm_disp = disp / mean_disp

    grad_disp_x, grad_disp_y = get_forward_difference(norm_disp, data_format)
    grad_img_x, grad_img_y = get_forward_difference(img, data_format)

    weight_x = tf.exp(-tf.reduce_mean(tf.abs(grad_img_x), axis=channel_axis, keepdims=True))
    weight_y = tf.exp(-tf.reduce_mean(tf.abs(grad_img_y), axis=channel_axis, keepdims=True))

    smoothness_x = tf.abs(grad_disp_x) * weight_x
    smoothness_y = tf.abs(grad_disp_y) * weight_y
    smoothness_loss = tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    return smoothness_loss


def normalize_motion_map(residual_translation, translation, data_format):
    """
    Normalizes a residual motion map by the motion map's norm.
    Arguments:
      residual_translation: residual translation of the shape  [B, H, W, 3] (or [B, 3, H, W])
      translation: camera translation of the shape  [B, H, W, 3] (or [B, 3, H, W])
      data_format: channels_first or channels_last
    """
    if data_format == "channels_first":
      axis = 1
    else:
      axis = -1
    translation_norm = tf.norm(translation, axis=axis, keepdims=True)
    return residual_translation / translation_norm


def l1_smoothness(tensor, data_format):
    """
    Calculates L1 (total variation) smoothness loss of a tensor.
    The goal of the loss to enforce continuity on the rigid body motion

    Arguments:
      tensor: A tensor to be smoothed, of shape [B, H, W, C] or [B,C, H, W] .
      data_format: channels_first or channels_last
    Returns:
      The total variation loss.
    """
    if data_format == "channels_first":
        axis = [2, 3]
    else:
        axis = [1, 2]
    epsilon = 1e-24 # need to avoid instability in training for intrinsics matrix
    # Note this is not exactly gradient but also has continuity over the boundaries (see tf.roll docs)
    tensor_dx = tensor - tf.roll(tensor, shift=1, axis=axis[1])
    tensor_dy = tensor - tf.roll(tensor, shift=1, axis=axis[0])
    return tf.reduce_mean(tf.sqrt(tf.square(tensor_dx) + tf.square(tensor_dy) + epsilon))


def sqrt_sparsity(motion_map, data_format):
    """
    A regularizer that encourages sparsity.

    This regularizer penalizes nonzero values. Close to zero it behaves like an L1
    regularizer, and far away from zero its strength decreases. The scale that
    distinguishes "close" from "far" is the mean value of the absolute of
    `motion_map`.

    Args:
        motion_map: A tf.Tensor of shape [B, H, W, 3] or [B, 3, H, W]

    Returns:
        A scalar the regularizer to be added to the training loss.
    """
    if data_format == "channels_first":
      spatial_axis = [2, 3]
    else:
      spatial_axis = [1, 2]
    epsilon = 1e-24 # add to avoid division by 0
    tensor_abs = tf.abs(motion_map)
    mean = tf.stop_gradient(tf.reduce_mean(tensor_abs, axis=spatial_axis, keepdims=True))
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return tf.reduce_mean(mean * tf.sqrt(1.0 + tensor_abs / (mean + epsilon)))

