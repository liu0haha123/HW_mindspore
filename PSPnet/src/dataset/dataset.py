import mindspore
import mindspore.dataset as ds
import os
import glob
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import mindspore.common.dtype as mstype
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import matplotlib.pyplot as plt

def get_data_list(data_list_file):
    with open(data_list_file, mode='r') as f:
        lines = []
        for line in f.readlines():
            lines.append(line.rstrip("\n"))
        return lines


# 定义数据集的读取方式，不包含增广
EXTENSIONS = [".jpg", ".jpeg", ".png"]


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def read_test_list(path):
    with open(path, "r", encoding="utf-8") as G:
        lines = G.readlines()
        all_list = [l.rstrip("\n") for l in lines]

    return all_list


def read_ade20k_image_label_list(data_dir, mode):
    if mode == 'train':
        data_sub = 'training'
    else:
        data_sub = 'validation'

    images_filename_proto = data_dir + '/ADE/images/' + data_sub + '/*.jpg'
    images = sorted(glob.glob(images_filename_proto))

    labels_filename_proto = data_dir + '/ADE/annotations/' + data_sub + '/*.png'
    labels = sorted(glob.glob(labels_filename_proto))

    # for just checking they are corresponded.
    for i in range(len(images)):
        if images[i].split('.jpg')[0].split('/')[-1] == labels[i].split('.png')[0].split('/')[-1]:
            continue

        print('< Error >', i, images[i], labels[i])

    return images, labels


class ADE20k():
    def __init__(self, root_path, num_classes, aug=True, mode="train"):
        super(ADE20k, self).__init__()
        self.data_dir = root_path
        self.mode = mode
        self.aug = aug
        self.num_classes = num_classes
        self.image_list, self.label_list = read_ade20k_image_label_list(self.data_dir, mode)
        self.min_scale = 0.5
        self.max_scale = 2.0
        self.image_mean = np.array([123.675, 116.280, 103.530])
        self.image_std = np.array([58.395, 57.120, 57.375])
        self.ignore_label = 255
        self.crop_size = 473

    def preprocess_(self, image, label):
        """SegDataset.preprocess_"""
        # bgr image

        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image.shape[0]), int(sc * image.shape[1])
        image_out = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]
        image_out = cv2.GaussianBlur(image_out, (5, 5), 0)
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def preprocess_eval(self, image, label):
        """preprocess for eval"""
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        h_off = int((h - self.crop_size) / 2)
        w_off = int((w - self.crop_size) / 2)
        image = image[h_off:h_off + self.crop_size, w_off:w_off + self.crop_size]
        label = label[h_off:h_off + self.crop_size, w_off:w_off + self.crop_size]

        image = (image - self.image_mean) / self.image_std
        image = image.transpose((2, 0, 1))
        return image, label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        filename_image = self.image_list[index]
        filename_label = self.label_list[index]

        image = cv2.imread(filename_image, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        label = cv2.imread(filename_label, cv2.IMREAD_GRAYSCALE)
        if self.mode == "train":
            image, label = self.preprocess_(image, label)
        else:
            image, label = self.preprocess_eval(image, label)

        return image, label


class VOC12Dataset():

    def __init__(self, root_path, num_classes, mode, aug):
        self.aug = aug
        self.num_classes = num_classes
        self.mode = mode
        self.min_scale = 0.5
        self.max_scale = 2.0
        self.image_mean = np.array([123.675, 116.280, 103.530])
        self.image_std = np.array([58.395, 57.120, 57.375])
        self.ignore_label = 255
        self.crop_size = 473

        if self.mode == "train" or self.mode == "eval":
            # 以标注图像为准
            self.images_root = os.path.join(root_path, "VOC2012", 'JPEGImages')
            self.labels_root = os.path.join(root_path, "VOC2012", 'SegmentationClass')
            self.filenames = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        else:
            print("请指定数据集划分")
        train_list = get_data_list(os.path.join(root_path, "VOC2012", 'ImageSets/Segmentation/train.txt'))
        val_list = get_data_list(os.path.join(root_path, "VOC2012", 'ImageSets/Segmentation/val.txt'))
        if (self.mode == "train"):
            self.filenames = train_list
        elif (self.mode == "eval"):
            self.filenames = val_list

    def __getitem__(self, index):

        filename = self.filenames[index]

        filename_image = self.images_root + "/" + str(filename) + '.jpg'
        image = cv2.imread(filename_image, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order

        filename_label = self.labels_root + "/" + str(filename) + '.png'
        label = cv2.imread(filename_label, cv2.IMREAD_GRAYSCALE)
        if self.mode == "train":
            image, label = self.preprocess_(image, label)
        else:
            image, label = self.preprocess_eval(image, label)

        return image, label

    def preprocess_(self, image, label):
        """SegDataset.preprocess_"""
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image.shape[0]), int(sc * image.shape[1])
        image_out = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]
        image_out = cv2.GaussianBlur(image_out, (5, 5), 0)
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def preprocess_eval(self, image, label):
        """preprocess for eval"""
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        h_off = int((h - self.crop_size) / 2)
        w_off = int((w - self.crop_size) / 2)
        image = image[h_off:h_off + self.crop_size, w_off:w_off + self.crop_size]
        label = label[h_off:h_off + self.crop_size, w_off:w_off + self.crop_size]

        image = (image - self.image_mean) / self.image_std
        image = image.transpose((2, 0, 1))
        return image, label

    def __len__(self):
        return len(self.filenames)


def get_dataset_VOC(num_classes, root_path, aug, mode, repeat, shard_num, shard_id, batch_size, num_workers=1):
    dataset_VOC = VOC12Dataset(root_path, num_classes, mode, aug)
    dataset_size = len(dataset_VOC)
    if mode=="train":
        data_set = ds.GeneratorDataset(source=dataset_VOC, column_names=["data", "label"],
                                   shuffle=True,
                                   num_shards=shard_num, shard_id=shard_id, num_parallel_workers=num_workers)
    else:
        data_set = ds.GeneratorDataset(source=dataset_VOC, column_names=["data", "label"],shuffle=False)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set, dataset_size


def get_dataset_ADE(num_classes, root_path, aug, mode, repeat, shard_num, shard_id, batch_size, num_workers=1):
    dataset_ADE = ADE20k(root_path=root_path, num_classes=num_classes, aug=aug, mode=mode)
    dataset_size = len(dataset_ADE)
    if mode== "train":
        data_set = ds.GeneratorDataset(source=dataset_ADE, column_names=["data", "label"],
                                   shuffle=True, num_shards=shard_num, shard_id=shard_id,
                                   num_parallel_workers=num_workers)
    else:
        data_set = ds.GeneratorDataset(source=dataset_ADE, column_names=["data", "label"],
                                   shuffle=False)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set, dataset_size

dataset_eval = VOC12Dataset(root_path="E:\\hw_ms\\data",num_classes=21,aug=False,mode="eval")
dataset_eval = ds.GeneratorDataset(source=dataset_eval, column_names=["data", "label"],shuffle=False)

for i,data in enumerate(dataset_eval.create_dict_iterator(output_numpy=True)):
    image = data["data"]
    label = data["label"]
    plt.subplot(2,1,1)
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.subplot(2,1,2)
    plt.imshow(label)
    plt.show()