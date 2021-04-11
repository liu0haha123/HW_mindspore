import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import mindspore.dataset.engine as de
import numpy as np
from mindspore.dataset.vision import c_transforms as C


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
    def __init__(self, base_dir, downsample_factor, data_augmentation, mode):
        super(Dataset_DIV2K, self).__init__()
        if mode == "train":
            self.filelist_label = sorted(
                [join(base_dir, "DIV2K800_sub", x) for x in listdir(os.path.join(base_dir, "data/DIV2K/DIV2K800_sub")) if
                 is_image_file(x)])
            LR_dir = join(base_dir, "data/DIV2K",("DIV2K800_sub_LRx" + str(downsample_factor)))
            self.filelist_img = sorted(
                [os.path.join(base_dir, LR_dir, x) for x in listdir(os.path.join(base_dir, LR_dir)) if
                 is_image_file(x)])

        elif mode == "valid":
            self.filelist_label = sorted(
                [join(base_dir,"data", "DIV2K_valid_HR", x) for x in listdir(os.path.join(base_dir, "data/DIV2K/DIV2K_train_HR")) if
                 is_image_file(x)])
            LR_dir = join(base_dir, "data/DIV2K",("DIV2K_valid_LR_X" + str(downsample_factor)))
            self.filelist_img = sorted(
                [os.path.join(base_dir, LR_dir, x) for x in listdir(os.path.join(base_dir, LR_dir)) if
                 is_image_file(x)])

        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        input = load_img(self.filelist_img[index])

        target = load_img(self.filelist_label[index])

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        input = np.transpose(input / 255.0, (2, 0, 1))
        target = np.transpose(target / 255.0, (2, 0, 1))
        return input, target

    def __len__(self):
        return len(self.filelist_img)


class Dataset_Flickr():
    def __init__(self, base_dir, downsample_factor, data_augmentation, mode, num_trainset):
        super(Dataset_Flickr, self).__init__()

        self.filelist_label = sorted(
            [join(x, base_dir, "Flickr2K_HR", x) for x in listdir(os.path.join(base_dir, "Flickr2K_HR")) if
             is_image_file(x)])
        LR_dir = join(base_dir, "Flickr2K_LR_bicubic", ("X" + str(downsample_factor)))
        self.filelist_img = sorted(
            [os.path.join(base_dir, LR_dir, x) for x in listdir(os.path.join(base_dir, LR_dir)) if
             is_image_file(x)])

        if mode == "train":
            self.filelist_img = self.filelist_img[:num_trainset]
            self.filelist_label = self.filelist_label[:num_trainset]
        else:
            self.filelist_img = self.filelist_img[num_trainset:]
            self.filelist_label = self.filelist_label[num_trainset:]
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        input = load_img(self.filelist_img[index])

        target = load_img(self.filelist_label[index])

        if self.data_augmentation:
            input, target, _ = augment(input, target)
        input = np.transpose(input / 255.0, (2, 0, 1))
        target = np.transpose(target / 255.0, (2, 0, 1))
        return input, target

    def __len__(self):
        return len(self.filelist_img)


def get_dataset_DIV2K(base_dir, downsample_factor, mode, aug, repeat, num_readers, shard_num, shard_id, batch_size):
    dataset_DIV2K = Dataset_DIV2K(base_dir, downsample_factor, data_augmentation=aug, mode=mode)
    data_set = de.GeneratorDataset(source=dataset_DIV2K, column_names=["input", "target"],
                                   shuffle=True, num_parallel_workers=num_readers,
                                   num_shards=shard_num, shard_id=shard_id)
    data_set = data_set.shuffle(buffer_size=batch_size * 10)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set


def get_dataset_Flickr(base_dir, downsample_factor, mode, aug, repeat, resize_shape, num_trainset, num_readers,
                       shard_num, shard_id,
                       num_parallel_calls, batch_size):
    dataset_Flickr = Dataset_Flickr(base_dir, downsample_factor, data_augmentation=aug, mode=mode,
                                    num_trainset=num_trainset)
    data_set = de.GeneratorDataset(source=dataset_Flickr, column_names=["input", "target"],
                                   shuffle=True, num_parallel_workers=num_readers,
                                   num_shards=shard_num, shard_id=shard_id)
    data_set = data_set.shuffle(buffer_size=batch_size * 10)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set
