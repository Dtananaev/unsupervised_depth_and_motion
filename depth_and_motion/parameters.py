#
# Author: Denis Tananaev
# Date: 19.02.2021
#

class Parameters(object):
    """
    The class contains experiment parameters.
    """

    def __init__(self):


        self.settings = {

            # The directory for checkpoints
            "dataset_dir": "/media/denis/SSD_A/ssai_dataset",
            "data_format": "channels_last", # channels_first not yet implemented
            "data_height": 128,
            "data_width":  448,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "depth_weight_decay": 0.0,
            "motion_weight_decay": 0.0,
            # The checkpoint related
            "checkpoints_dir": "log/checkpoints",
            "train_summaries": "log/summaries/train",
            "eval_summaries": "log/summaries/val",
            # Update tensorboard train images each step_summaries iterations
            "step_summaries": 100,  # to turn off make it None
            # General settings
            "seed": 2021,
            "max_epochs": 1000,
            # Training settings
            "clip_grad": 10.0, # clip gradient higher than val to this val # to turn off put None
            # If nonzero, motion fields will be unfrozen after motion_field_burnin_steps
            # steps. Over the first half of the motion_field_burnin_steps steps, the
            # motion fields will be zero. Then the ramp up is linear.
            # masks all values bigger than mean_sq_residual_translation
            "motion_auto_mask": True,
            "motion_field_burning_steps": 15,  # The value  (by default 20000 iteration (15 epochs))put None if want to turn off
            "normalize_scale": False, # normalize translation and rotation with respect to the depth mean
            # Stops gradient on the target depth when computing the depth
            # consistency loss.
            "target_depth_stop_gradient": True, # stop gradient on depth
            "max_video_idx":1, # number of videos with different intrinsics
        }
        self.settings["loss_weights"] = {
                'rgb_consistency': 1.0,
                'ssim': 1.5,
                'depth_consistency': 0.0,
                'depth_smoothing': 0.001,
                'rotation_cycle_consistency': 1e-3,
                'translation_cycle_consistency': 5e-2,
                'motion_smoothing': 1.0,
                'motion_sparsity': 0.4,
                }
        # Automatically defined during training parameters
        self.settings["train_size"] = None  # the size of train set
        self.settings["val_size"] = None  # the size of val set