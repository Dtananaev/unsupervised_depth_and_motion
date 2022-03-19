#
# Author: Denis Tananaev
# Date: 27.02.2021
#

import tensorflow as tf
import os
import numpy as np
from depth_and_motion.sfm_dataset import SfmDataset
from depth_and_motion.parameters import Parameters
from depth_and_motion.tools.training_tools import load_model, initialize_model, setup_gpu
from depth_and_motion.models.resnet18unet import ResNet18Unet
from depth_and_motion.models.intrinsicsnet import IntrinsicsNet
from depth_and_motion.models.motionnet import MotionNet
from depth_and_motion.summary_helpers import (
    train_summaries,
)
from depth_and_motion.loss_agregator import loss_agregator

import argparse
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure train runs on gpu 0


def initialize_training_models(param, train_dataset, resume):
    """
    This functions initializes or resumes training for depth and camera
    networks.
    """
    # Init model depth
    model_depth = ResNet18Unet(
        weight_decay=param["depth_weight_decay"],
        data_format=param["data_format"],
        name="ResNet18Unet",
    )
    initialize_model(
        model_depth,
        height=param["data_height"],
        width=param["data_width"],
        channels=3,
        data_format=param["data_format"],
    )
    start_epoch_depth, model_depth = load_model(
        param["checkpoints_dir"], model_depth, resume
    )
    # Init model motion
    model_motion = MotionNet(
        weight_decay=param["motion_weight_decay"],
        data_format=param["data_format"],
        auto_mask=param["motion_auto_mask"],
        name="MotionNet",
    )
    initialize_model(
        model_motion,
        height=param["data_height"],
        width=param["data_width"],
        channels=8,
        data_format=param["data_format"],
    )
    start_epoch_motion, model_motion = load_model(
        param["checkpoints_dir"], model_motion, resume
    )

    model_intrinsics = IntrinsicsNet(
        max_video_index=param["max_video_idx"],
        data_format=param["data_format"],
        name="IntrinsicsTable",
    )
    start_epoch_intrinsics, model_intrinsics = load_model(
        param["checkpoints_dir"], model_intrinsics, resume
    )

    # Check that we load checkpoint with the same epoch
    if (start_epoch_motion != start_epoch_depth) & (
        start_epoch_intrinsics != start_epoch_depth
    ):
        raise ValueError(
            f"Can't resume training inconsinstent checkpoints for depth {start_epoch_depth} and camera {start_epoch_motion} epochs."
        )

    models_dict = {
        "model_depth": model_depth,
        "model_motion": model_motion,
        "model_intrinsics": model_intrinsics,
    }
    return models_dict, start_epoch_depth


@tf.function
def train_step(param, samples, models_dict, optimizer):
    if param["data_format"] == "channels_first":
        channels_axis = 1
    else:
        channels_axis = -1

    # The depth computed for first and second image
    loss = 0.0
    endpoints = {}
    with tf.GradientTape() as tape:
        # Compute depth
        first_img, second_img, video_idx = samples
        rgb_stack = tf.concat([first_img, second_img], axis=0)
        predicted_depth = models_dict["model_depth"](rgb_stack, training=True)
        endpoints["predicted_depth"] = tf.split(predicted_depth, 2, axis=0)
        endpoints["rgb"] = tf.split(rgb_stack, 2, axis=0)

        motion_features = [
            tf.concat(
                [endpoints["rgb"][0], endpoints["predicted_depth"][0]],
                axis=channels_axis,
            ),
            tf.concat(
                [endpoints["rgb"][1], endpoints["predicted_depth"][1]],
                axis=channels_axis,
            ),
        ]

        motion_input = tf.concat(motion_features, axis=0)
        flipped_motion_input = tf.concat(motion_features[::-1], axis=0)
        # Unlike `rgb_stack`, here we stacked the frames in reverse order along the
        # Batch dimension. By concatenating the two stacks below along the channel
        # axis, we create the following tensor:
        #
        #         Channel dimension (3)
        #   _                                 _
        #  |  Frame1-s batch | Frame2-s batch  |____Batch
        #  |_ Frame2-s batch | Frame1-s batch _|    dimension (0)
        #
        # When we send this tensor to the motion prediction network, the first and
        # second halves of the result represent the camera motion from Frame1 to
        # Frame2 and from Frame2 to Frame1 respectively. Further below we impose a
        # loss that drives these two to be the inverses of one another
        # (cycle-consistency).
        pairs = tf.concat([motion_input, flipped_motion_input], axis=channels_axis)
        rot, trans, residual_translation = models_dict["model_motion"](
            pairs, training=True
        )
        intrinsics_mat, distortion = models_dict["model_intrinsics"](
            (first_img, video_idx), training=True
        )

        if param["motion_field_burning_steps"] is not None:
            burnin_steps = tf.cast(
                param["motion_field_burning_steps"]
                * (param["train_size"] / param["batch_size"]),
                tf.float32,
            )
            step = tf.cast(optimizer.iterations, tf.float32)
            residual_translation *= tf.clip_by_value(
                2 * step / burnin_steps - 1, 0.0, 1.0
            )

        endpoints["residual_translation"] = tf.split(residual_translation, 2, axis=0)
        endpoints["background_translation"] = tf.split(trans, 2, axis=0)
        endpoints["rotation"] = tf.split(rot, 2, axis=0)
        endpoints["intrinsics_mat"] = [intrinsics_mat] * 2
        endpoints["distortion"] = [distortion] * 2

        losses, loss_endpoints = loss_agregator(
            endpoints,
            param["data_format"],
            param["loss_weights"],
            param["normalize_scale"],
            param["target_depth_stop_gradient"],
        )
        for key in losses:
            loss += losses[key]
        # Get L2 losses for weight decay
        # For all optimizers without decoupled weight decay
        # For adamW and FTRL this is not needed

        loss += tf.add_n(models_dict["model_depth"].losses)
        loss += tf.add_n(models_dict["model_motion"].losses)

        trainable_variables = (
            models_dict["model_depth"].trainable_variables
            + models_dict["model_motion"].trainable_variables
            + models_dict["model_intrinsics"].trainable_variables
        )
        gradients = tape.gradient(loss, trainable_variables)
        if param["clip_grad"] is not None:
            gradients = [
                tf.clip_by_norm(g, clip_norm=param["clip_grad"]) for g in gradients
            ]
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        train_outputs = {
            "losses": losses,
            "endpoints": endpoints,
            "loss_endpoints": loss_endpoints,
            "rot_scale": models_dict["model_motion"].rot_scale,
            "trans_scale": models_dict["model_motion"].trans_scale,
        }

        return train_outputs



def train(resume=False):
    setup_gpu()
    # General parameters
    param = Parameters().settings
    # Random seed for experiments reproduction
    tf.random.set_seed(param["seed"])

    input_shape = (param["data_width"], param["data_height"])
    train_dataset = SfmDataset(
        param["dataset_dir"],
        "train.datalist",
        param["batch_size"],
        param["data_format"],
        input_shape,
        shuffle=True,
    )
    param["train_size"] = train_dataset.num_samples

    models_dict, start_epoch = initialize_training_models(param, train_dataset, resume)
    models_dict["model_depth"].summary()
    models_dict["model_motion"].summary()
    optimizer = tf.keras.optimizers.Adam(param["learning_rate"])



    model_path = os.path.join(param["checkpoints_dir"], "{model}-{epoch:04d}")
    for epoch in range(start_epoch, param["max_epochs"]):
        save_depth_dir = model_path.format(
            model=models_dict["model_depth"].name, epoch=epoch
        )
        save_camera_dir = model_path.format(
            model=models_dict["model_motion"].name, epoch=epoch
        )
        save_intr_dir = model_path.format(
            model=models_dict["model_intrinsics"].name, epoch=epoch
        )
        for train_samples in tqdm(
            train_dataset.dataset,
            desc=f"Epoch {epoch}",
            total=train_dataset.num_it_per_epoch,
        ):
            train_outputs = train_step(param, train_samples, models_dict, optimizer)
            train_summaries(train_outputs, optimizer, param)

        # Save all
        models_dict["model_depth"].save(save_depth_dir)
        models_dict["model_motion"].save(save_camera_dir)
        models_dict["model_intrinsics"].save(save_intr_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN.")
    parser.add_argument(
        "--resume", type=lambda x: x, nargs="?", const=True, default=False
    )
    args = parser.parse_args()
    train(resume=args.resume)