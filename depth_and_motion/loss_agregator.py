#
# Author: Denis Tananaev
# Date: 27.02.2021
#

import tensorflow as tf
from depth_and_motion.regularization_losses import edge_aware_smoothness_loss, sqrt_sparsity, l1_smoothness, normalize_motion_map
from depth_and_motion.losses import rgbd_and_motion_consistency_loss

def loss_agregator(endpoints, data_format, weights, normalize_scale, target_depth_stop_gradient):
    losses = {}
    output_endpoints = {}


    if data_format == "channels_first":
        channels_axis = 1
    else:
        channels_axis = -1
    # Concat images over batch size
    rgb_stack = tf.concat(endpoints['rgb'], axis=0)
    flipped_rgb_stack = tf.concat(endpoints['rgb'][::-1], axis=0)

    predicted_depth_stack = tf.concat(endpoints['predicted_depth'], axis=0)
    flipped_predicted_depth_stack = tf.concat(endpoints['predicted_depth'][::-1], axis=0)

    residual_translation = tf.concat(endpoints['residual_translation'], axis=0)
    flipped_residual_translation = tf.concat(endpoints['residual_translation'][::-1], axis=0)    

    intrinsics_mat = tf.concat(endpoints['intrinsics_mat'], axis=0)
    distortion = tf.concat(endpoints['distortion'], axis=0)

    background_translation = tf.concat(endpoints['background_translation'], axis=0)
    flipped_background_translation = tf.concat(endpoints['background_translation'][::-1], axis=0)

    rotation = tf.concat(endpoints['rotation'], axis=0)
    flipped_rotation = tf.concat(endpoints['rotation'][::-1], axis=0)


    if normalize_scale:
        mean_depth = tf.reduce_mean(predicted_depth_stack)
        predicted_depth_stack /= mean_depth
        flipped_predicted_depth_stack /= mean_depth
        background_translation /= mean_depth
        flipped_background_translation /= mean_depth
        residual_translation /= mean_depth
        flipped_residual_translation /= mean_depth

    translation = residual_translation + background_translation
    flipped_translation = flipped_residual_translation + flipped_background_translation


    disp = 1.0 / predicted_depth_stack
    predicted_depth_stack = tf.squeeze(predicted_depth_stack, axis=channels_axis)
    flipped_predicted_depth_stack = tf.squeeze(flipped_predicted_depth_stack, axis=channels_axis)


    if target_depth_stop_gradient:
        flipped_predicted_depth_stack = tf.stop_gradient(flipped_predicted_depth_stack)


    loss_endpoints = rgbd_and_motion_consistency_loss(rgb_stack,
                                                      predicted_depth_stack,
                                                      flipped_rgb_stack,
                                                      flipped_predicted_depth_stack,
                                                      rotation,
                                                      translation,
                                                      flipped_rotation,
                                                      flipped_translation,
                                                      intrinsics_mat,
                                                      data_format=data_format,
                                                      distortion_coeff=distortion,
                                                      )

    normalized_trans = normalize_motion_map(residual_translation, translation, data_format)
    # Regularization losses
    losses['depth_smoothing'] = weights["depth_smoothing"] * edge_aware_smoothness_loss(disp, rgb_stack, data_format)
    losses["motion_smoothing"] = weights["motion_smoothing"] * l1_smoothness(normalized_trans, data_format)
    losses["motion_sparsity"] = weights["motion_sparsity"] * sqrt_sparsity(normalized_trans, data_format)


    losses['depth_consistency'] =  weights["depth_consistency"] * loss_endpoints['depth_error']
    losses['rgb_consistency'] = weights["rgb_consistency"] * loss_endpoints['rgb_error']
    losses['ssim'] =  weights["ssim"] * loss_endpoints['ssim_error']
    losses['rotation_cycle_consistency'] = weights["rotation_cycle_consistency"]  * loss_endpoints['rotation_error']
    losses['translation_cycle_consistency'] =  weights["translation_cycle_consistency"]  * loss_endpoints['translation_error']

    output_endpoints["disp"] = tf.split(disp, 2, 0)
    output_endpoints['depth_proximity_weight'] = tf.split(loss_endpoints['depth_proximity_weight'], 2, 0)
    output_endpoints['frame1_closer_to_camera'] = tf.split(loss_endpoints['frame1_closer_to_camera'], 2, 0)
    output_endpoints['trans'] = translation
    output_endpoints['inv_trans'] = flipped_translation
    return losses, output_endpoints