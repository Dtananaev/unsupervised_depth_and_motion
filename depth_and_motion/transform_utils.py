#
# Author: Denis Tananaev
# Date: 09.03.2021
#

import tensorflow as tf
import numpy as np


def euler_to_mat(z, y, x):
  """
  Converts euler angles to rotation matrix.
  Arguments:
    z: rotation angle along z axis (in radians) -- size = [B, ...]
    y: rotation angle along y axis (in radians) -- size = [B, ...]
    x: rotation angle along x axis (in radians) -- size = [B, ...]
  Returns:
    Rotation matrix corresponding to the euler angles, with shape [B, ..., 3, 3].
  """
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to ... x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros_like(z)
  ones = tf.ones_like(z)

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=-1)
  rotz_2 = tf.concat([sinz, cosz, zeros], axis=-1)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=-1)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=-2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=-1)
  roty_2 = tf.concat([zeros, ones, zeros], axis=-1)
  roty_3 = tf.concat([-siny, zeros, cosy], axis=-1)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=-2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=-1)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=-1)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=-1)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=-2)

  return tf.matmul(tf.matmul(xmat, ymat), zmat)


def matrix_from_angles(rot):
  """
  Create a rotation matrix from a triplet of rotation angles.

  Arguments:
    rot: Tensor of shape [..., 3], where the last dimension is the rotation
      angles, along x, y, and z.

  Returns:
    mat: tensor of shape [..., 3, 3], where the last two dimensions are the
    rotation matrix.
  """
  rank = tf.rank(rot)
  # Swap the two last dimensions
  perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)
  mat = tf.transpose(euler_to_mat(rot[..., 2], rot[..., 1], rot[..., 0]) , perm) 

  return  mat 


def unstacked_matrix_from_angles(rx, ry, rz):
  """Create an unstacked rotation matrix from rotation angles.

  Args:
    rx: A tf.Tensor of rotation angles abound x, of any shape.
    ry: A tf.Tensor of rotation angles abound y (of the same shape as x).
    rz: A tf.Tensor of rotation angles abound z (of the same shape as x).

  Returns:
    A 3-tuple of 3-tuple of Tensors of the same shape as x, representing the
    respective rotation matrix. The small 3x3 dimensions are unstacked into a
    tuple to avoid tensors with small dimensions, which bloat the TPU HBM
    memory. Unstacking is one of the recommended methods for resolving the
    problem.
  """
  angles = [-rx, -ry, -rz]
  sx, sy, sz = [tf.sin(a) for a in angles]
  cx, cy, cz = [tf.cos(a) for a in angles]
  m00 = cy * cz
  m10 = (sx * sy * cz) - (cx * sz)
  m20 = (cx * sy * cz) + (sx * sz)
  m01 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m21 = (cx * sy * sz) - (sx * cz)
  m02 = -sy
  m12 = sx * cy
  m22 = cx * cy
  return ((m00, m01, m02), (m10, m11, m12), (m20, m21, m22))


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
  """
  Composes two transformations, each has a rotation and a translation.

  Arguments:
    rot_mat1: tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec1: tensor of shape [..., 3] representing translation vectors.
    rot_mat2: tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec2: tensor of shape [..., 3] representing translation vectors.

  Returns:
    A tuple of 2 Tensors, representing rotation matrices and translation
    vectors, of the same shapes as the input, representing the result of
    applying rot1, trans1, rot2, trans2, in succession.
  """
  # Building a 4D transform matrix from each rotation and translation, and
  # multiplying the two, we'd get:
  #
  # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
  # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
  #
  # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
  # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
  # total translation is R2*t1 + t2.
  r2r1 = tf.matmul(rot_mat2, rot_mat1)
  r2t1 = tf.matmul(rot_mat2, tf.expand_dims(trans_vec1, -1))
  r2t1 = tf.squeeze(r2t1, axis=-1)
  return r2r1, r2t1 + trans_vec2
