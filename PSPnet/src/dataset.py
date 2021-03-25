import mindspore
import mindspore.dataset as ds
import os
import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import glob
from random import shuffle
import numpy as np
import math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

num_classes_VOC = 21
num_classes_ADE = 150
EXTENSIONS = [".jpg", ".jpeg", ".png"]
VOC_image_size = [473, 473, 3]

def load_image(file):
    return Image.open(file)

def letterbox_image(image, label, size):
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
    def __init__(self, root_path,num_classes, aug= True,mode="train"):
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
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(VOC_image_size[0]), int(VOC_image_size[1]), self.num_classes + 1))

        # jpg = np.transpose(np.array(jpg), [2, 0, 1]) / 255
        jpg = np.array(jpg) / 255
        return jpg, png, seg_labels


class VOC12Dataset():

    def __init__(self, root_path, num_classes, aug=True, mode="train"):
        self.mode = mode
        self.aug = aug
        self.num_classes = num_classes
        if self.mode == "train" or self.mode == "dev":
            self.images_root = os.path.join(root_path, "VOC2012traindev", 'JPEGImages')
            self.labels_root = os.path.join(root_path, "VOC2012traindev", 'SegmentationClass')
            self.filenames = [image_basename(f)
                              for f in os.listdir(self.labels_root) if is_image(f)]
        else:
            print("请指定数据集划分")

        if (self.mode == "train"):
            self.filenames = self.filenames[:2600]
        elif (self.mode == "eval"):
            self.filenames = self.filenames[2600:]
        elif (self.mode == "test"):
            self.filenames = self.filenames

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

        filename = self.filenames[index]

        with open(self.images_root + "/" + str(filename) + '.jpg', 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(self.labels_root + "/" + str(filename) + '.png', 'rb') as f:
            label = load_image(f).convert('P')

        if self.aug:
            jpg, png = self.get_random_data(image, label, (int(VOC_image_size[0]), int(VOC_image_size[1])))
        else:
            jpg, png = letterbox_image(image, label, (int(VOC_image_size[1]), int(VOC_image_size[0])))

        # 从文件中读取图像
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        # 转化成one_hot的形式
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(VOC_image_size[0]), int(VOC_image_size[1]), self.num_classes + 1))

        #jpg = np.transpose(np.array(jpg), [2, 0, 1]) / 255
        jpg = np.array(jpg)/255
        return jpg, png, seg_labels

    def __len__(self):
        return len(self.filenames)

def create_dataset(dataset_name,dataset_path,mode,config,repeat_num=1):
    """
    创建语义分割的数据集.

    Args:
        dataset_name（string）：指定数据集名称
        dataset_path (string): 指定数据集路径
        mode :  dataset is used for train or eval.
        config: configuration
        repeat_num (int): The repeat times of dataset. Default: 1.
    Returns:
        Dataset.
    """


    device_id = 0
    device_num = 1
    if config.platform == "GPU":
        if config.run_distribute:
            from mindspore.communication.management import get_rank, get_group_size
            device_id = get_rank()
            device_num = get_group_size()
    elif config.platform == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
    if mode == "train":
        do_shuffle = True
        if dataset_name == "VOC2012":
            dataset = VOC12Dataset(root_path=dataset_path, num_classes=num_classes_VOC, mode=mode)
        elif dataset_name == "ADE20K":
            dataset = ADE20k(root_path=dataset_path, num_classes=num_classes_ADE, mode=mode)
    else:
        do_shuffle = False
        if dataset_name == "VOC2012":
            dataset = VOC12Dataset(root_path=dataset_path, aug=False,num_classes=num_classes_VOC, mode=mode)
        elif dataset_name == "ADE20K":
            dataset = ADE20k(root_path=dataset_path, aug=False,num_classes=num_classes_ADE, mode=mode)
    if device_num == 1 or not (mode=="train"):
        ds = de.GeneratorDataset(dataset_path, num_parallel_workers=4, shuffle=do_shuffle)
    else:
        ds = de.GeneratorDataset(dataset_path, num_parallel_workers=4, shuffle=do_shuffle,
                               num_shards=device_num, shard_id=device_id)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 100


    # define map operations
    random_horizontal_flip_op = C.RandomHorizontalFlip(device_id / (device_id + 1))
    resize_op = C.Resize((resize_height, resize_width))
    normalize_op = C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    change_swap_op = C.HWC2CHW()
    trans = []
    if mode=="train":
        trans += [random_horizontal_flip_op]

    trans += [resize_op, normalize_op, change_swap_op]

    #type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=trans)
    ds = ds.map(input_columns="image", operations=trans)


    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(config.batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds

"""
dataset = ADE20k(data_dir="/home/hoo/ms_dataset/ADE", mode="train")

ADE_data = ds.GeneratorDataset(dataset, ["input", "label"], shuffle=True)

for data in ADE_data.create_dict_iterator():
    print(type(data["input"]))
    dataset_test = VOC12Dataset(root_path="/home/hoo/ms_dataset/VOC2012", num_classes=21,mode="train")
dataset = ds.GeneratorDataset(dataset_test, ["image", "label","seg_label"], shuffle=False)

for data in dataset.create_dict_iterator():
    plt.subplot(1,2,1)
    plt.imshow(data['image'].asnumpy())
    plt.subplot(1, 2, 2)
    plt.imshow(data["label"].asnumpy())
    plt.show()

"""
