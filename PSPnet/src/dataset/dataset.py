import mindspore
import mindspore.dataset as ds
import os
import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

# 定义数据集的读取方式，不包含增广
num_classes_VOC = 21
num_classes_ADE = 150
EXTENSIONS = [".jpg", ".jpeg", ".png"]
VOC_image_size = [473, 473, 3]


def load_image(file):
    return Image.open(file)


def letterbox_image(image, label, size):
    # 随机裁剪已有大小的图像不涉及其他增广
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
    return new_image, new_label


def rand(a, b):
    return np.random.rand() * (b - a) + a


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
    def __init__(self, root_path, num_classes, aug=True, mode="train"):
        super(ADE20k, self).__init__()
        self.data_dir = root_path
        self.mode = mode
        self.aug = aug
        self.num_classes = num_classes
        self.image_list, self.label_list = read_ade20k_image_label_list(self.data_dir, mode)
        assert len(self.image_list) > 0, 'No images are found.'
        print('Database has %d images.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # flip image or not
        flip = rand(0, 1) < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand(0, 1) < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand(0, 1) < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def __getitem__(self, index):
        filename_image = self.image_list[index]
        filename_label = self.label_list[index]

        image = load_img(filename_image).convert('RGB')
        label = load_img(filename_label).convert('P')

        if self.aug:
            jpg, png = self.get_random_data(image, label, (int(VOC_image_size[0]), int(VOC_image_size[1])))
        else:
            jpg, png = letterbox_image(image, label, (int(VOC_image_size[1]), int(VOC_image_size[0])))

        # 从文件中读取图像
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        # 转化成one_hot的形式
        # seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        # seg_labels = seg_labels.reshape((int(VOC_image_size[0]), int(VOC_image_size[1]), self.num_classes + 1))

        jpg = np.transpose(np.array(jpg), [2, 0, 1]) / 255
        return jpg, png


class VOC12Dataset():

    def __init__(self, num_classes, lst_path, aug=True):
        self.aug = aug
        self.num_classes = num_classes
        self.img_path = []
        self.label_path = []
        with open(lst_path) as f:
            lines = f.readlines()
            print('number of samples:', len(lines))
            for l in lines:
                img_path, label_path = l.strip().split(' ')
                self.img_path.append(img_path)
                self.label_path.append(label_path)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        # resize image
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # flip image or not
        flip = rand(0, 1) < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand(0, 1) < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand(0, 1) < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def __getitem__(self, index):

        img = self.img_path[index]
        label = self.label_path[index]
        with open(img, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(label, 'rb') as f:
            label = load_image(f).convert('P')

        if self.aug:
            jpg, png = self.get_random_data(image, label, (int(VOC_image_size[0]), int(VOC_image_size[1])))
        else:
            jpg, png = letterbox_image(image, label, (int(VOC_image_size[1]), int(VOC_image_size[0])))

        # 从文件中读取图像
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        # 转化成one_hot的形式
        # seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        # seg_labels = seg_labels.reshape((int(VOC_image_size[0]), int(VOC_image_size[1]), self.num_classes + 1))
        # 默认的读入格式是NHWC注意转换成NCWH
        jpg = np.transpose(np.array(jpg), [2, 0, 1]) / 255

        return jpg, png

    def __len__(self):
        return len(self.img_path)


def get_dataset_VOC(num_classes, lst_path, aug, repeat, shard_num, shard_id, batch_size):
    dataset_VOC = VOC12Dataset(num_classes, lst_path, aug)
    data_set = ds.GeneratorDataset(source=dataset_VOC, column_names=["data", "label"],
                                   shuffle=True,
                                   num_shards=shard_num, shard_id=shard_id)
    data_set = data_set.shuffle(buffer_size=batch_size * 10)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set


def get_dataset_ADE(num_classes, root_path, aug, mode, repeat, num_readers, shard_num, shard_id, batch_size):
    dataset_VOC = ADE20k(root_path=root_path, num_classes=num_classes, aug=aug, mode=mode)
    data_set = de.GeneratorDataset(dataset=dataset_VOC, columns_list=["data", "label"],
                                   shuffle=True,
                                   num_shards=shard_num, shard_id=shard_id)
    data_set = data_set.shuffle(buffer_size=batch_size * 10)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat)
    return data_set

