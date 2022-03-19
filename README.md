#  Unsupervised Depth And Motion Estimation

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Dtananaev/unsupervised_depth_and_motion/blob/master/LICENSE.md) 

This is reimplementation of the paper [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/abs/2010.16404)

## Installation
For ubuntu 18.04 install necessary dependecies:
```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```
Create virtual environment and activate it:
```
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```
Upgrade pip tools:
```
pip install --upgrade pip
```
Install tensorflow 2.0  (for more details check the tensofrolow install tutorial: [tensorflow](https://www.tensorflow.org/install/pip))
```
pip install --upgrade tensorflow-gpu
```
Clone this repository and then install it:
```
cd path_to_this_folder
pip install -e .
```

## Dataset creation

The dataset folder structure should be the next:
``` bash
    dataset
    ├── 00001   # The sequence folder
    │   ├──00000.jpg  # the images extracted from the video
    │   └── ..  
    └── ...
```
In order to create dataset list apply:
```
cd depth_and_motion
python create_datalist.py --dataset_dir <path_to_the_dataset_folder>
```
This will creates the pair of images and the corresponding number for sequence folder list for self-supervised training.
Note: Each sequnce folder assumed to be separate video, therefore it will learn its own intrinsics
In order to run training:
```
python train.py
```

