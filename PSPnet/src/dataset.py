import mindspore
import mindspore.dataset as ds
import numpy as np
import os
import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageOps

num_classes_ADE = 150
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



"""
def find_data(dataset="ADE20k"):
    if dataset == "ADE20k":
        img_mean = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)  # RGB, SBD/Pascal VOC.
        num_classes = 150
    else:
        raise ValueError("Unknown database %s" % dataset)

    return img_mean, num_classes
"""



def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def read_ade20k_image_label_list(data_dir, mode):
    if mode == 'train':
        data_sub = 'training'
    else:
        data_sub = 'validation'

    images_filename_proto = data_dir + '/images/' + data_sub + '/*.jpg'
    images = sorted(glob.glob(images_filename_proto))

    labels_filename_proto = data_dir + '/annotations/' + data_sub + '/*.png'
    labels = sorted(glob.glob(labels_filename_proto))

    # for just checking they are corresponded.
    for i in range(len(images)):
        if images[i].split('.jpg')[0].split('/')[-1] == labels[i].split('.png')[0].split('/')[-1]:
            continue

        print('< Error >', i, images[i], labels[i])

    return images, labels


class ADE20k():
    def __init__(self, data_dir, mode="train"):
        super(ADE20k, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.num_classes = num_classes_ADE
        self.image_list, self.label_list = read_ade20k_image_label_list(self.data_dir, mode)
        assert len(self.image_list) > 0, 'No images are found.'
        print('Database has %d images.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        filename_image = self.image_list[index]
        filename_label = self.label_list[index]

        image = load_img(filename_image).convert('RGB')
        label = load_img(filename_label).convert('P')

        return (image, label)



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

"""
dataset = ADE20k(data_dir="/home/hoo/ms_dataset/ADE", mode="train")

ADE_data = ds.GeneratorDataset(dataset, ["input", "label"], shuffle=True)

for data in ADE_data.create_dict_iterator():
    print(type(data["input"]))

dataset_test = VOC12Dataset(root_path="/home/hoo/ms_dataset/VOC2012", mode="train")
dataset = ds.GeneratorDataset(dataset_test, ["image", "label"], shuffle=False)

"""
