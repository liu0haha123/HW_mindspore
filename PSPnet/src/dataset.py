import mindspore
import mindspore.dataset as ds
import numpy as np
import os
from PIL import Image

EXTENSIONS = [".jpg", ".jpeg", ".png"]


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path(root, basename, extension):
    return os.path.join(root, '{basename}{extension}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def read_test_list(path):
    with open(path, "r", encoding="utf-8") as G:
        lines = G.readlines()
        all_list = [l.rstrip("\n") for l in lines]

    return all_list


class VOC12Dataset():

    def __init__(self, root_path, mode="train"):
        self.mode = mode
        if self.mode == "train" or self.mode == "dev":
            self.images_root = os.path.join(root_path, "VOC2012traindev", 'JPEGImages')
            self.labels_root = os.path.join(root_path, "VOC2012traindev", 'SegmentationClass')
            self.filenames = [image_basename(f)
                              for f in os.listdir(self.labels_root) if is_image(f)]
        elif self.mode == "test":
            self.images_root = os.path.join(root_path, "VOC2012test", 'JPEGImages')
            self.filenames = read_test_list(
                os.path.join(root_path, "VOC2012test", "ImageSets", "Segmentation", "test.txt"))
        else:
            print("请指定数据集划分")

        if (self.mode == "train"):
            self.filenames = self.filenames[:2600]
        elif (self.mode == "eval"):
            self.filenames = self.filenames[2600:]
        elif (self.mode == "test"):
            self.filenames = self.filenames

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "dev":
            filename = self.filenames[index]

            with open(self.images_root + "/" + str(filename) + '.jpg', 'rb') as f:
                image = load_image(f).convert('RGB')
            with open(self.labels_root + "/" + str(filename) + '.png', 'rb') as f:
                label = load_image(f).convert('P')

            return (image, label)
        else:
            filename = self.filenames[index]

            with open(self.images_root + "/" + str(filename) + '.jpg', 'rb') as f:
                image = load_image(f).convert('RGB')

            return (image, None)

    def __len__(self):
        return len(self.filenames)


dataset_test = VOC12Dataset(root_path="/home/hoo/ms_dataset/VOC2012", mode="train")

dataset = ds.GeneratorDataset(dataset_test, ["image", "label"], shuffle=False)
