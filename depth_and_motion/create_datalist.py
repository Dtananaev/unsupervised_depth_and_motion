#
# Author: Denis Tananaev
# Date: 18.02.2021
#
import os
import glob
import argparse
import numpy as np
from depth_and_motion.tools.io_file import save_dataset_list


class CreateUnsupervisedSfmDatasetList:
    """
    The class gets as input dataset folder with random youtube images
    and creates the pairs lists of data
    Arguments:
        dataset_dir: directory with data
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def remove_prefix(self, data_path, prefix):
        data_path = data_path[len(prefix) + 1 :]
        return data_path


    def create_dataset_file(self, data_list_name="train.datalist"):

        sequences = os.path.join(self.dataset_dir,  "*")
        sequences_list = sorted(glob.glob(sequences))
        dataset_list = []

        for idx, seq in enumerate(sequences_list):
            images_string = os.path.join(seq, "*.jpg")
            images_list = sorted(glob.glob(images_string))
            images_list = np.asarray([self.remove_prefix(x, self.dataset_dir) for x in images_list])

            for row in range(1, len(images_list)-1):
                data_sample = [images_list[i] for i in range(row - 1, row + 1)]
                data_sample.append(str(idx))
                dataset_list.append(";".join(data_sample))
        dataset_filename = os.path.join(self.dataset_dir, data_list_name)
        save_dataset_list(dataset_filename, dataset_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetFileCreator.")
    parser.add_argument(
        "--dataset_dir", type=str, help="creates .datalist file", default="/media/denis/SSD_A/ssai_dataset"
    )
    args = parser.parse_args()

    data_creator = CreateUnsupervisedSfmDatasetList(args.dataset_dir)
    data_creator.create_dataset_file()