import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import mindspore
import mindspore.dataset as ds
import matplotlib.pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class Dataset_DIV2K():
    def __init__(self, base_dir, patch_size, upscale_factor, downsample_factor, data_augmentation, mode):
        super(Dataset_DIV2K, self).__init__()
        if mode == "train":
            self.filelist_img = sorted(
                [join(x, base_dir, "DIV2K_train_HR", x) for x in listdir(os.path.join(base_dir, "DIV2K_train_HR")) if
                 is_image_file(x)])
            LR_dir = join(base_dir, ("DIV2K_train_LR_X" + str(downsample_factor)))
            self.filelist_label = sorted(
                [os.path.join(base_dir, LR_dir, x) for x in listdir(os.path.join(base_dir, LR_dir)) if
                 is_image_file(x)])

        elif mode == "valid":
            self.filelist_img = sorted(
                [join(x, base_dir, "DIV2K_valid_HR", x) for x in listdir(os.path.join(base_dir, "DIV2K_train_HR")) if
                 is_image_file(x)])
            LR_dir = join(base_dir, ("DIV2K_valid_LR_X" + str(downsample_factor)))
            self.filelist_label = sorted(
                [os.path.join(base_dir, LR_dir, x) for x in listdir(os.path.join(base_dir, LR_dir)) if
                 is_image_file(x)])

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        input = load_img(self.filelist_img[index])

        target = load_img(self.filelist_img[index])

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        return input, target

    def __len__(self):
        return len(self.filelist_img)


base_dir = "/home/hoo/ms_dataset/DIV2K"
dataset_test = Dataset_DIV2K(base_dir="/home/hoo/ms_dataset/DIV2K", patch_size=40, upscale_factor=1,
                             downsample_factor=2, data_augmentation=True, mode="train")
dataset = ds.GeneratorDataset(dataset_test, ["input", "target"], shuffle=True)

for data in dataset.create_dict_iterator():
    plt.imshow(data['target'].asnumpy().squeeze(), cmap=plt.cm.gray)
    plt.show()