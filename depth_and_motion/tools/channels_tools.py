#
# Author: Denis Tananaev
# Date: 26.02.2021
#
import tensorflow as tf
import numpy as np


def to_channels_first_tf(tensor):
  """
  The function gets tensor of [N,H,W,C] (or H,W,C) and returns tensor of [N,C,H,W] (or C,H,W).
  Tensorflow function.                                                  
  """
  shape_len = len(tf.shape(tensor)) #tf.size
  if shape_len == 3:
        out = tf.transpose(tensor, [2, 0, 1])
  elif shape_len == 4:
      out = tf.transpose(tensor, [0, 3, 1, 2])
  else:
      raise ValueError(f"The shape of the tensor should be 3 or 4 dimensions but it is {shape_len}")
  return out


def to_channels_first_np(tensor):
  """
  The function gets tensor of [N,H,W,C] (or H, W, C) and returns tensor of [N,C,H,W] (or C, H, W).
  Numpy function.                                                  
  """
  shape_len = len(tensor.shape)
  if shape_len == 3:
        out = np.transpose(tensor, (2, 0, 1))
  elif shape_len == 4:
      out = np.transpose(tensor, (0, 3, 1, 2))
  else:
      raise ValueError(f"The shape of the tensor should be 3 or 4 dimensions but it is {shape_len}")
  return out


def to_channels_last_tf(tensor):
  """The function gets tensor of [N,C,H,W] (or C,H,W) and returns tensor of [N,H,W,C] (or H,W,C)
  Tensorflow function.
  """
  shape_len = len(tf.shape(tensor))
  if shape_len == 3:
        out = tf.transpose(tensor, [1, 2, 0])
  elif shape_len == 4:
      out = tf.transpose(tensor, [0, 2, 3, 1])
  else:
      raise ValueError(f"The shape of the tensor should be 3 or 4 dimensions but it is {shape_len}")
  out = tf.transpose(tensor, )
  return out   


def to_channels_last_np(tensor):
  """The function gets tensor of [N,C,H,W] (or C,H,W) and returns tensor of [N,H,W,C] (or H,W,C)
  Numpy function.                                                  
  """
  shape_len = len(tensor.shape)
  if shape_len == 3:
        out = np.transpose(tensor, (1, 2, 0))
  elif shape_len == 4:
      out = np.transpose(tensor, (0, 2, 3, 1))
  else:
      raise ValueError(f"The shape of the tensor should be 3 or 4 dimensions but it is {shape_len}")
  return out
