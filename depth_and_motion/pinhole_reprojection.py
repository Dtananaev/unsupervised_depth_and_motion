#
# Author: Denis Tananaev
# Date: 12.03.2021
#

import tensorflow as tf
from depth_and_motion.transform_utils import matrix_from_angles
from depth_and_motion.tools.tensors_tools import replace_nonfinite
from depth_and_motion.tools.channels_tools import to_channels_last_tf


def quadraric_distortion_scale(distortion_coefficient, r_squared):
  """Calculates a quadratic distortion factor given squared radii.

  The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
  `distortion_coefficient` is negative (barrel distortion), the distorted radius
  is only monotonically increasing only when
  `r_squared` < r_squared_max = -1 / (3 * distortion_coefficient).

  Args:
    distortion_coefficient: A tf.Tensor of a floating point type. The rank can
      be from zero (scalar) to r_squared's rank. The shape of
      distortion_coefficient will be appended by ones until the rank equals that
      of r_squared.
    r_squared: A tf.Tensor of a floating point type, containing
      (x/z)^2 + (y/z)^2. We use r_squared rather than r to avoid an unnecessary
      sqrt, which may introduce gradient singularities. The non-negativity of
      r_squared only enforced in debug mode.

  Returns:
    A tf.Tensor of r_squared's shape, the correction factor that should
    multiply the projective coordinates (x/z) and (y/z) to apply the
    distortion.
  """
  return 1 + distortion_coefficient * r_squared


def quadratic_inverse_distortion_scale(distortion_coefficient, distorted_r_squared, newton_iterations=4):
  """Calculates the inverse quadratic distortion function given squared radii.

  The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
  `distortion_coefficient` is negative (barrel distortion), the distorted radius
  is monotonically increasing only when
  r < r_max = sqrt(-1 / (3 * distortion_coefficient)).
  max_distorted_r_squared is obtained by calculating the distorted_r_squared
  corresponding to r = r_max, and the result is
  max_distorted_r_squared = - 4 / (27.0 * distortion_coefficient)

  Args:
    distortion_coefficient: A tf.Tensor of a floating point type. The rank can
      be from zero (scalar) to r_squared's rank. The shape of
      distortion_coefficient will be appended by ones until the rank equals that
      of r_squared.
    distorted_r_squared: A tf.Tensor of a floating point type, containing
      (x/z)^2 + (y/z)^2. We use distorted_r_squared rather than distorted_r to
      avoid an unnecessary sqrt, which may introduce gradient singularities.
      The non-negativity of distorted_r_squared is only enforced in debug mode.
    newton_iterations: Number of Newton-Raphson iterations to calculate the
      inverse distprtion function. Defaults to 5, which is on the high-accuracy
      side.

  Returns:
    A tf.Tensor of distorted_r_squared's shape, containing the correction
    factor that should multiply the distorted the projective coordinates (x/z)
    and (y/z) to obtain the undistorted ones.
  """
  c = 1.0  # c for Correction
  # Newton-Raphson iterations for solving the inverse function of the
  # distortion.
  for _ in range(newton_iterations):
    c = (1.0 -
         (2.0 / 3.0) * c) / (1.0 + 3 * distortion_coefficient *
                             distorted_r_squared * c * c) + (2.0 / 3.0) * c
  return c


def meshgrid(width, height):
    """
    The function creates meshgrid from width and height
    Arguments: 
        width: width of the input
        height: height of the input
    Return:
        grid: meshgrid of the shape [3, heigth, width]
    """
    grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=-1)
    return  tf.cast(grid, tf.float32)


def pixels_to_camera(intrinsic_mat, pixel_coords):
    """
    The function produces 3D point cloud from depth, intrinsics and pixels_grid
    Arguments:
        intrinsic_mat: intrinsic matrix of the shape [B, 3, 3]
        pixel_coords: pixels coordinates of the shape  [3, heigth, width]
    Returns:
        normalized_grid: normalized camera coordinates (z=1) of the shape [B, 3, H, W] 
    """
    intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)
    normalized_grid = tf.einsum('bij,jhw->bihw', intrinsic_mat_inv, pixel_coords)
    return normalized_grid


def camera_to_pixels(intrinsic_mat, point_cloud):
  """
  The function gets intrinsics and point cloud and projects point cloud to the image
  Arguments:
    intrinsic_mat: intrinsic matrix of the shape [B, 3, 3]
    point_cloud: 3d point cloud in camera coord system of the shape [B, 3, H, W]
  Returns:
    unnormalized_pcoords: pixels coordinates  of the shape [B, 3, H, W] here to get the normalized pixels
    use next: 
    x, y, z = tf.unstack(unnormalized_pcoords, axis=1)
    x_pixels = x/z
    y_pixels = y/z
    z - depth for each pixel
  """
  # Project the transformed point cloud back to the camera plane.
  unnormalized_pcoords = tf.einsum('bij,bjhw->bihw', intrinsic_mat, point_cloud)
  return unnormalized_pcoords


def get_valid_reprojection_mask(x_pixels, y_pixels, z):
    """
        Reprojected depth and corresponding pixels mapping x, y, z all shapes [B, H, W]
      Returns:
        Valid pixels mask for evaluation of the shape [B, H, W]
    """
    _, height, width = tf.unstack(tf.shape(x_pixels))
    x_max = tf.cast(width - 1, tf.float32)
    y_max = tf.cast(height - 1, tf.float32)
    # Boundary mask
    mask_x = tf.math.logical_and(x_pixels >= 0.0, x_pixels < x_max)
    mask_y = tf.math.logical_and( y_pixels >= 0.0, y_pixels < y_max)
    # Finite mask
    finite_mask =  tf.math.logical_and(tf.math.is_finite(x_pixels), tf.math.is_finite(y_pixels))
    z_positive = z > 0.0
    valid_reprojection_mask = mask_x & mask_y & finite_mask & z_positive

    # Mask out non relevant pixels
    x_pixels = replace_nonfinite(x_pixels)
    y_pixels = replace_nonfinite(y_pixels)
    x_pixels = tf.clip_by_value(x_pixels, 0.0, x_max)
    y_pixels = tf.clip_by_value(y_pixels, 0.0, y_max)
    return x_pixels, y_pixels, valid_reprojection_mask


def reprojection(depth, translation, rotation_angles, intrinsic_mat, distortion_coeff=None):
    """
    The reprojection from frame 1 to frame 2 given depth from frame 1 translation and rotation
    from frame 1 to frame 2 intrinsics matrix and distortion coeff
    Arguments:
        depth: depth from the frame1 of the shape [B, H, W]
        translation: A Tensor of shape [B, 3] or [B, H, W, 3] representing a
        translation vector for the entire image or for every pixel respectively.
        rotation_angles: A Tensor of shape [B, 3] or [B, H, W, 3] representing a
        set of rotation angles for the entire image or for every pixel
        intrinsic_mat: intrinsics matrix of the shape [B, 3, 3]
        distortion_coeff: a scalar for the radial distortion if None or 0.0 will not be computed
    Returns:
        Reprojected depth and corresponding pixels mapping x, y, z all shapes [B, H, W]
    """

    if translation.shape.ndims == 2:
        translation = tf.expand_dims(tf.expand_dims(translation, 1), 1)

    _, height, width = tf.unstack(tf.shape(depth))
    pixels_grid = meshgrid(width, height)

    normalized_pixels = pixels_to_camera(intrinsic_mat, pixels_grid)

    if distortion_coeff is not None:
        radii_squared = tf.reduce_sum(tf.square(normalized_pixels[:, :2, :, :]), axis=1)
        undistortion_factor = quadratic_inverse_distortion_scale(distortion_coeff, radii_squared)
        undistortion_factor = tf.stack([undistortion_factor, undistortion_factor, tf.ones_like(undistortion_factor)],axis=1)
        normalized_pixels *= undistortion_factor

    rot_mat = matrix_from_angles(rotation_angles)
    # We have to treat separately the case of a per-image rotation vector and a
    # per-image rotation field, because the broadcasting capabilities of einsum
    # are limited.
    if rotation_angles.shape.ndims == 2:
        # The calculation here is identical to the one in inverse_warp above.
        # Howeverwe use einsum for better clarity. Under the hood, einsum performs
        # the reshaping and invocation of BatchMatMul, instead of doing it manually,
        # as in inverse_warp.
        pcoords = tf.einsum('bij,bjhw,bhw->bihw', rot_mat, normalized_pixels, depth)
    elif rotation_angles.shape.ndims == 4:
        # We push the H and W dimensions to the end, and transpose the rotation
        # matrix elements (as noted above).
        rot_mat = tf.transpose(rot_mat, [0, 3, 4, 1, 2])
        pcoords = tf.einsum('bijhw,bjhw,bhw->bihw', rot_mat, normalized_pixels, depth)


    pcoords += tf.transpose(translation, [0, 3, 1, 2])
    x, y, z = tf.unstack(pcoords, axis=1)
    x /= z
    y /= z

    if distortion_coeff is not None:
        scale = quadraric_distortion_scale(distortion_coeff, tf.square(x) + tf.square(y))
        x *= scale
        y *= scale
    
    pcoords = tf.stack([x, y, tf.ones_like(z)], axis=1)
    pcoords = camera_to_pixels(intrinsic_mat, pcoords)
    x, y, _ = tf.unstack(pcoords, axis=1)
    x, y, valid_reprojection_mask = get_valid_reprojection_mask(x, y, z)
    return x, y, z, valid_reprojection_mask


def apply_reprojection(depth, translation, rotation_angles, intrinsic_mat, data_format, distortion_coeff=None):
      """
      The reprojection from frame 1 to frame 2 given depth from frame 1 translation and rotation
      from frame 1 to frame 2 intrinsics matrix and distortion coeff
      Arguments:
          depth: depth from the frame1 of the shape [B, H, W]
          translation: A Tensor of shape [B, 3] or [B, H, W, 3] representing a
          translation vector for the entire image or for every pixel respectively.
          rotation_angles: A Tensor of shape [B, 3] or [B, H, W, 3] representing a
          set of rotation angles for the entire image or for every pixel
          intrinsic_mat: intrinsics matrix of the shape [B, 3, 3]
          distortion_coeff: a scalar for the radial distortion if None or 0.0 will not be computed
      Returns:
          Reprojected depth and corresponding pixels mapping x, y, z all shapes [B, H, W]
      """

      if translation.shape.ndims == 4 and data_format == "channels_first":
          translation = to_channels_last_tf(translation)

      if rotation_angles.shape.ndims == 4 and data_format == "channels_first":
          rotation_angles = to_channels_last_tf(rotation_angles)      
      
      return reprojection(depth, translation, rotation_angles, intrinsic_mat, distortion_coeff)