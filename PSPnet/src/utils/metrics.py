import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from os.path import join
import numpy as np
from PIL import Image
from mindspore.ops import operations as P
from mindspore import dtype as mstype

# 定义损失函数


class SoftmaxCrossEntropyLoss(nn.Cell):
    # 带忽略标签的交叉熵
    def __init__(self, num_cls, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(
            labels_int, self.num_cls, self.on_value, self.off_value
        )
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


# 这部分用于后处理
# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def calc_semantic_segmentation_confusion(pred_labels, gt_labels, num_classes):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = num_classes
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()   # (168960, )
        gt_label = gt_label.flatten()   # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.  极度巧妙 × class_nums 正好使得每个ij能够对应.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2)\
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


# def generate_matrix(gt_image, pre_image, num_class=cfg.DATASET_Num_Class):
#     mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
#     混淆矩阵计算原始版
#     label = num_class * gt_image[mask].astype('int') + pre_image[mask]
#     # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
#     count = np.bincount(label, minlength=num_class ** 2)
#     confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
#     return confusion_matrix

# PA
def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc

# MIoU


def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)  # 跳过0值求mean,shape:[21]
    return MIoU

# MPA


def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    pa = Pixel_Accuracy(confusion)
    mpa = Pixel_Accuracy_Class(confusion)
    miou = Mean_Intersection_over_Union(confusion)

    return {
        'pa': pa,
        "mpa": mpa,
        'miou': miou,
    }

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo
