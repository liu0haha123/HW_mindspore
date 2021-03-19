import random
import math
import numpy as np
import numbers
import collections
import cv2
from mindspore import Tensor
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C_trans
import mindspore.dataset.transforms.py_transforms as PY_trans
import mindspore.dataset.vision.py_transforms as PY_trans_V
import mindspore.dataset.vision.c_transforms as C_trans_V
from mindspore.dataset.vision import Inter,Border
# 核心思想 利用mindspore的已有算子，保证功能对应，不保证实现完全相同

class Compose(object):
    def __init__(self,seg_transform_C,seg_transform_Py):
        self.seg_transform_C = seg_transform_C
        self.seg_transform_Py = seg_transform_Py
    def __call__(self,dataset):
        compose_C = C_trans.Compose(self.seg_transform_C)
        dataset = dataset.map(operations=compose_C,input_columns=["img","label"])
        compose_py = PY_trans.Compose(self.seg_transform_Py)
        dataset = dataset.map(operations=compose_py, input_columns=["img", "label"])
        return dataset

class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a mindspore Tensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("need data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = Tensor(image.transpose((2, 0, 1)),dtype=mindspore.float32)
        label = Tensor(label,dtype=mindspore.int64)
        return image, label

def Normalize(mean,std):
    normalize_op = C_trans_V.Normalize(mean=mean,std=std)
    return normalize_op

def decode():
    decode_op = C_trans_V.Decode()
    return decode_op

def Resize_image(size):
    # 原始现中 image的插值方法是线性插值
    Resize_op = C_trans_V.Resize(size=size[::-1],interpolation=Inter.LINEAR)
    return Resize_op

def Resize_label(size):
    # 原始现中 label的插值方法是最近邻插值
    Resize_op = C_trans_V.Resize(size=size[::-1],interpolation=Inter.NEAREST)
    return Resize_op

def Crop_Center(crop_h,crop_w):
    # 随机的中心裁剪 方式通用
    Crop_ops = C_trans_V.CenterCrop(size=(crop_h,crop_w))
    return Crop_ops

def RandCrop_image(crop_h,crop_w,padding):
    # image 专用，因为填充值不同
    Crop_ops = C_trans_V.RandomCrop(size=(crop_h,crop_w),fill_value=padding)
    return Crop_ops

def RandCrop_label(crop_h,crop_w,ignore_index):
    # label 专用，因为填充值不同
    Crop_ops = C_trans_V.RandomCrop(size=(crop_h,crop_w),fill_value=ignore_index)
    return Crop_ops

def RandScale(scale,aspect_ratio):
    # 随机放缩输入图像
    if random.random()<aspect_ratio:
        Scale_op = C_trans_V.Rescale(scale,shift=-1.0)
    else:
        Scale_op = C_trans_V.Rescale(0.0,0.0)

    return Scale_op


def RandomHorizontalFlip(p):
    RandomHorizontalFlip_op = C_trans_V.RandomHorizontalFlip(p)
    return RandomHorizontalFlip_op

def RandomVerticalFlip(p):
    RandomVerticalFlip_op = C_trans_V.RandomVerticalFlip(p)
    return RandomVerticalFlip_op

def Rotate_image(rotate_angle,padding,p):
    # image专用
    if random.random()<p:
        rotate_op = C_trans_V.RandomRotation(rotate_angle,resample=Inter.LINEAR, expand=True,fill_value=padding)
    else:
        rotate_op = C_trans_V.RandomRotation(0,resample=Inter.LINEAR, expand=True,fill_value=padding)
    return rotate_op

def Rotate_label(rotate_angle,ignore_index,p):
    # label 专用
    if random.random()<p:
        rotate_op = C_trans_V.RandomRotation(rotate_angle,resample=Inter.NEAREST, expand=True,fill_value=ignore_index)
    else:
        rotate_op = C_trans_V.RandomRotation(0,resample=Inter.NEAREST, expand=True,fill_value=ignore_index)
    return rotate_op

class RandomGaussianBlur(object):
    # 无法直接应用于框架中，需要在读取数据时处理
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
