#
# Author: Denis Tananaev
# Date: 12.03.2021
#
import tensorflow as tf
from depth_and_motion import resampler
from depth_and_motion.pinhole_reprojection import apply_reprojection
from depth_and_motion import transform_utils


def rgbd_consistency_loss(warp_x, warp_y, frame1transformed_depth, reproj_valid_mask, frame1_rgb, frame2_depth, frame2_rgb, data_format):
    """
    The function computes rgb consistency loss
    Arguments:
    frame1transformed_depth: the depth transformed from frame1 to frame2
    frame1_rgb: frame 1 rgb image of the shape [B, H, W, 3] (or [B,3,H,W])
    frame2_depth: depth predicted for frame2 with the shape [B, H, W]
    frame2_rgb: frame 2 rgb image of the shape [B, H, W, 3] (or [B,3,H,W])
    """

    if data_format=="channels_first":
        channels_axis = 1
    else:
        channels_axis= -1
    # Concat depth for frame 2 to bilinearly resample for comparison with frame1transformed depth
    frame2_rgbd = tf.concat([frame2_rgb, tf.expand_dims((frame2_depth), channels_axis)], axis=channels_axis)
    # Resampling
    frame2_rgbd_resampled = resampler.resampler_with_unstacked_warp(frame2_rgbd, warp_x, warp_y, safe=False)
    # Split images from depths
    frame2rgb_resampled, frame2depth_resampled = tf.split(frame2_rgbd_resampled, [3, 1], axis=channels_axis)
    frame2depth_resampled = tf.squeeze(frame2depth_resampled, axis=channels_axis)

    # We penalize inconsistencies between the two frames' depth maps only if the
    # transformed depth map (of frame 1) falls closer to the camera than the
    # actual depth map (of frame 2). This is intended for avoiding penalizing
    # points that become occluded because of the transform.
    # So what about depth inconsistencies where frame1's depth map is FARTHER from
    # the camera than frame2's? These will be handled when we swap the roles of
    # frame 1 and 2 (more in https://arxiv.org/abs/1904.04998)
    frame1_closer_to_camera = tf.cast( tf.logical_and(reproj_valid_mask, tf.less(frame1transformed_depth, frame2depth_resampled)), tf.float32)

    # Compute depth consistency
    frames_l1_diff = tf.abs(frame2depth_resampled - frame1transformed_depth)
    depth_error = tf.reduce_mean(tf.math.multiply_no_nan(frames_l1_diff, frame1_closer_to_camera))

    # Compute photometric loss
    frames_rgb_l1_diff = tf.abs(frame2rgb_resampled - frame1_rgb)
    rgb_error = tf.math.multiply_no_nan(frames_rgb_l1_diff, tf.expand_dims(frame1_closer_to_camera, channels_axis))
    rgb_error = tf.reduce_mean(rgb_error)

    # Compute ssim
    # We generate a weight function that peaks (at 1.0) for pixels where  the
    # depth difference is less than its standard deviation across the frame, and
    # fall off to zero otherwise. This function is used later for weighing the
    # structural similarity loss term. We only want to demand structural
    # similarity for surfaces that are close to one another in the two frames.
    depth_sq_diff = tf.square(frame2depth_resampled - frame1transformed_depth)
    depth_error_second_moment = _weighted_average(depth_sq_diff, frame1_closer_to_camera, data_format) + 1e-4
    depth_proximity_weight = tf.math.multiply_no_nan(depth_error_second_moment / (depth_sq_diff + depth_error_second_moment), tf.cast(reproj_valid_mask, tf.float32))
    # If we don't stop the gradient training won't start. The reason is presumably
    # that then the network can push the depths apart instead of seeking RGB
    # consistency.
    depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)
    ssim_error, avg_weight = weighted_ssim(frame2rgb_resampled, frame1_rgb, depth_proximity_weight, c1=float('inf'), c2=9e-6)
    ssim_error_mean = tf.reduce_mean(tf.math.multiply_no_nan(ssim_error, avg_weight))

    endpoints = {
        'depth_error': depth_error,
        'rgb_error': rgb_error,
        'ssim_error': ssim_error_mean,
        'depth_proximity_weight': depth_proximity_weight,
        'frame1_closer_to_camera': frame1_closer_to_camera
    }
    return endpoints


def motion_field_consistency_loss(warp_x, warp_y, frame1_closer_to_camera, rotation1, translation1, rotation2, translation2, data_format):
    """Computes a cycle consistency loss between two motion maps.

    Given two rotation and translation maps (of two frames), and a mapping from
    one frame to the other, this function assists in imposing that the fields at
    frame 1 represent the opposite motion of the ones in frame 2.

    In other words: At any given pixel on frame 1, if we apply the translation and
    rotation designated at that pixel, we land on some pixel in frame 2, and if we
    apply the translation and rotation designated there, we land back at the
    original pixel at frame 1.

    Args:
    frame1transformed_pixelx: A tf.Tensor of shape [B, H, W] representing the
        motion-transformed x-location of each pixel in frame 1.
    frame1transformed_pixely: A tf.Tensor of shape [B, H, W] representing the
        motion-transformed y-location of each pixel in frame 1.
    mask: A tf.Tensor of shape [B, H, W, 2] expressing the weight of each pixel
        in the calculation of the consistency loss.
    rotation1: A tf.Tensor of shape [B, 3] representing rotation angles.
    translation1: A tf.Tensor of shape [B, H, W, 3] representing translation
        vectors.
    rotation2: A tf.Tensor of shape [B, 3] representing rotation angles.
    translation2: A tf.Tensor of shape [B, H, W, 3] representing translation
        vectors.

    Returns:
    A dicionary from string to tf.Tensor, with the following entries:
        rotation_error: A tf scalar, the rotation consistency error.
        translation_error: A tf scalar, the translation consistency error.

    """
    translation2resampled = resampler.resampler_with_unstacked_warp(translation2, tf.stop_gradient(warp_x), tf.stop_gradient(warp_y), safe=False)
    rotation1field = tf.broadcast_to(_expand_dims_twice(rotation1, -2), tf.shape(translation1))
    rotation2field = tf.broadcast_to(_expand_dims_twice(rotation2, -2), tf.shape(translation2))
    rotation1matrix = transform_utils.matrix_from_angles(rotation1field)
    rotation2matrix = transform_utils.matrix_from_angles(rotation2field)

    rot_unit, trans_zero = transform_utils.combine(rotation2matrix, translation2resampled, rotation1matrix, translation1)
    eye = tf.eye(3, batch_shape=tf.shape(rot_unit)[:-2])

    # We normalize the product of rotations by the product of their norms, to make
    # the loss agnostic of their magnitudes, only wanting them to be opposite in
    # directions. Otherwise the loss has a tendency to drive the rotations to
    # zero.
    rot_error = tf.reduce_mean(tf.square(rot_unit - eye), axis=(3, 4))
    rot1_scale = tf.reduce_mean(tf.square(rotation1matrix - eye), axis=(3, 4))
    rot2_scale = tf.reduce_mean(tf.square(rotation2matrix - eye), axis=(3, 4))
    rot_error /= (1e-24 + rot1_scale + rot2_scale)
    rotation_error = tf.reduce_mean(rot_error)

    def norm(x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    # Here again, we normalize by the magnitudes, for the same reason.
    translation_error = tf.reduce_mean(tf.math.multiply_no_nan(frame1_closer_to_camera, norm(trans_zero) / (1e-24 + norm(translation1) + norm(translation2resampled))))


    endpoints = {
        'rotation_error': rotation_error,
        'translation_error': translation_error
    }
    return endpoints

def rgbd_and_motion_consistency_loss(frame1_rgb, frame1_depth, frame2_rgb, frame2_depth, rotation1, translation1, rotation2, translation2, intrinsics, data_format, distortion_coeff=None):
    """
    The function computes consisstency loss
    Arguments:
    frame1_rgb: frame1_rgb of the shape [N, H, W, C] (or [N, C, H, W])
    frame1_depth: frame1_depth of the shape [N, H, W]
    frame2_rgb: frame2_rgb of the shape [N, H, W, C] (or [N, C, H, W])
    frame2_depth: frame2_depth of the shape [N, H, W]
    rotation1: rotation from frame1 to frame2 of the shape [N, 3] or [N, H, W, 3] (or [N, 3, H, W])
    translation1: translation from frame1 to frame2 of the shape [N, 3] or [N, H, W, 3] (or [N, 3, H, W])
    rotation2: rotation from frame2 to frame1 of the shape [N, 3] or [N, H, W, 3] (or [N, 3, H, W])
    translation2: translation from frame2 to frame1 of the shape [N, 3] or [N, H, W, 3] (or [N, 3, H, W])
    intrinsics: intrinsics matrix of the shape [B, 3, 3]
    data_format: channels_first or channels_last
    distortion_coeff: scalar for radial distortion first coeff
    """

    # Output mappimg for warping depth1 transformed to frame2 coord and valid pixels mask
    warp_x, warp_y, frame1transformed_depth, reproj_valid_mask  = apply_reprojection(frame1_depth, translation1, rotation1, intrinsics, data_format, distortion_coeff)
    endpoints = rgbd_consistency_loss(warp_x, warp_y, frame1transformed_depth, reproj_valid_mask, frame1_rgb, frame2_depth, frame2_rgb, data_format)
    mask = endpoints['frame1_closer_to_camera']
    endpoints.update(motion_field_consistency_loss(warp_x, warp_y, mask, rotation1, translation1, rotation2, translation2, data_format))
    return endpoints



def _weighted_average(x, w, data_format, epsilon=1.0):
    if data_format == "channels_first":
        spatial_axis = (2, 3)
    else:
        spatial_axis = (1, 2)

    weighted_sum = tf.reduce_sum(x * w, axis=spatial_axis, keepdims=True)
    sum_of_weights = tf.reduce_sum(w, axis=spatial_axis, keepdims=True)
    return weighted_sum / (sum_of_weights + epsilon)



def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
  """Computes a weighted structured image similarity measure.

  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.

  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.

  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
  if c1 == float('inf') and c2 == float('inf'):
    raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                     'likely unintended.')
  weight = tf.expand_dims(weight, -1)
  average_pooled_weight = _avg_pool3x3(weight)
  weight_plus_epsilon = weight + weight_epsilon
  inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

  def weighted_avg_pool3x3(z):
    wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
    return wighted_avg * inverse_average_pooled_weight

  mu_x = weighted_avg_pool3x3(x)
  mu_y = weighted_avg_pool3x3(y)
  sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
  sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
  sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
  if c1 == float('inf'):
    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
  elif c2 == float('inf'):
    ssim_n = 2 * mu_x * mu_y + c1
    ssim_d = mu_x**2 + mu_y**2 + c1
  else:
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  result = ssim_n / ssim_d
  return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight


def _avg_pool3x3(x):
  return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

def _expand_dims_twice(x, dim):
  return tf.expand_dims(tf.expand_dims(x, dim), dim)
