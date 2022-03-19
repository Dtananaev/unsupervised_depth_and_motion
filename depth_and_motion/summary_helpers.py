#
# Author: Denis Tananaev
# Date: 04.04.2020
#

import tensorflow as tf
from depth_and_motion.tools.tensors_tools import normalize_zero_one
from tensorboard.plugins.mesh import summary as mesh_summary



def train_summaries(train_out, optimizer, param_settings):
    """
    Visualizes  the train outputs in tensorboards
    """
    intr_mat = tf.concat(train_out["endpoints"]["intrinsics_mat"], 0)
    distortion = tf.concat(train_out["endpoints"]["distortion"], 0)
    data_format = param_settings["data_format"]
    writer = tf.summary.create_file_writer(param_settings["train_summaries"])
    with writer.as_default():
        # Losses
        losses = train_out["losses"]
        with tf.name_scope("Training losses"):
            for key in losses:
                tf.summary.scalar(key, losses[key], step=optimizer.iterations)
        with tf.name_scope("Training scales"):
            tf.summary.scalar("rot_scale", train_out["rot_scale"], step=optimizer.iterations)
            tf.summary.scalar("trans_scale", train_out["trans_scale"], step=optimizer.iterations)

        # Show images
        if (
            param_settings["step_summaries"] is not None
            and optimizer.iterations % param_settings["step_summaries"] == 0
        ):

            # Show Inputs
            with tf.name_scope("1-Inputs"):
                tf.summary.image("1.First images", train_out["endpoints"]['rgb'][0], step=optimizer.iterations)
                tf.summary.image("2.Second images", train_out["endpoints"]['rgb'][1], step=optimizer.iterations)
            with tf.name_scope("2-Disp"):
                tf.summary.image(f"First disp", normalize_zero_one(train_out["loss_endpoints"]['disp'][0], data_format=data_format), step=optimizer.iterations)
                tf.summary.image(f"Second disp", normalize_zero_one(train_out["loss_endpoints"]['disp'][1], data_format=data_format), step=optimizer.iterations)
            with tf.name_scope("3-Depth proximity weight"):
                tf.summary.image(f"First proximity weight",  tf.expand_dims(train_out["loss_endpoints"]['depth_proximity_weight'][0], axis=-1), step=optimizer.iterations)
                tf.summary.image(f"Second proximity weight",   tf.expand_dims(train_out["loss_endpoints"]['depth_proximity_weight'][1], axis=-1), step=optimizer.iterations)
            with tf.name_scope("4- frame1 close to camera"):
                tf.summary.image(f"First frame1 close to camera",  tf.expand_dims(train_out["loss_endpoints"]['frame1_closer_to_camera'][0], axis=-1), step=optimizer.iterations)
                tf.summary.image(f"Second frame1 close to camera",   tf.expand_dims(train_out["loss_endpoints"]['frame1_closer_to_camera'][1], axis=-1), step=optimizer.iterations)
            with  tf.name_scope("5-Translation"):
                tf.summary.image(f"Trans",  normalize_zero_one(tf.abs(train_out["loss_endpoints"]['trans']), data_format), step=optimizer.iterations)
                tf.summary.image(f"Inv trans",  normalize_zero_one(tf.abs(train_out["loss_endpoints"]['inv_trans']), data_format), step=optimizer.iterations)
            with tf.name_scope("6-Residual translataion"):
                tf.summary.image("1.First residual", normalize_zero_one(tf.abs(train_out["endpoints"]['residual_translation'][0]), data_format), step=optimizer.iterations)
                tf.summary.image("2.Second residual", normalize_zero_one(tf.abs(train_out["endpoints"]['residual_translation'][1]), data_format), step=optimizer.iterations)                
            with tf.name_scope("Distributions"):
                tf.summary.histogram("Predicted depth", tf.concat([train_out["endpoints"]["predicted_depth"]], 0), step=optimizer.iterations)
                tf.summary.histogram("residual_translation", tf.concat([train_out["endpoints"]["residual_translation"]], 0), step=optimizer.iterations)
                tf.summary.histogram("background_translation", tf.concat([train_out["endpoints"]["background_translation"]], 0), step=optimizer.iterations)
                tf.summary.histogram("rotation", tf.concat([train_out["endpoints"]["rotation"]], 0), step=optimizer.iterations)

            #with tf.name_scope("Point cloud"):
            #    mesh_summary.mesh(name="Point cloud", vertices=point_cloud, colors=point_cloud_colors, step=optimizer.iterations)
