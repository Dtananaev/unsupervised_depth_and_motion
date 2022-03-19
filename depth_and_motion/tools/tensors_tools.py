#
# Author: Denis Tananaev
# Date: 26.02.2021
#

import tensorflow as tf
from depth_and_motion.tools.channels_tools import to_channels_first_tf, to_channels_last_tf

def normalize_zero_one(tensor, data_format):
      if data_format =="channels_first":
            spatial_axis = [2,3]
      else:
            spatial_axis = [1,2]
      max = tf.reduce_max(tensor, axis=spatial_axis, keepdims=True)
      min = tf.reduce_min(tensor, axis=spatial_axis, keepdims=True)
      tensor = (tensor - min) / (max - min)
      return tensor
      
def resize_tensor(tensor, shape, data_format, method="nearest"):
      if data_format == "channels_first":
            tensor = to_channels_last_tf(tensor)
            tensor =  tf.image.resize(tensor, shape, method)
            tensor = to_channels_first_tf(tensor)
      else:
            tensor =  tf.image.resize(tensor, shape, method)
      return tensor


def get_forward_difference(tensor, data_format):

      if data_format =="channels_first":
            grad_x =tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
            grad_y = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
      else:
            grad_x = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
            grad_y = tensor[:, :-1, :, :] - tensor[:, 1:, :, :]

      return grad_x, grad_y


def replace_nonfinite(tensor):
  """The function gets tensor and replace all inf and NaN values with zeros.
                                                    
  Args:
    tensor: input tensor
  Returns:
    tensor: output tensor with all values finite
  """    
  tensor = tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))
  return tensor

def inverse(tensor):
  """The function gets tensor and teturns 1/tensor.
                                                    
  Args:
    tensor: input tensor
  Returns:
    inverse: inversed tensor
  """    
  inverse = tf.divide(tf.ones_like(tensor), tensor)
  return inverse

def log10(tensor):
  """ The function gets tensor computes log10 of it.
  Tensorflow doesn't support log10.
  We evaluate log 10 by apply logarithms rule:
  log_b(a) = log_x(a)/log_x(b)
                                                    
  Args:
    tensor: input tensor
  Returns:
      log_10_tensor: the log10 of input tensor
  """    
  numerator = tf.math.log(tensor)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  log_10_tensor = numerator / denominator
  return log_10_tensor
