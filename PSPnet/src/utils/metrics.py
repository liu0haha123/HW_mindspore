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
    def __init__(self, num_cls=21, ignore_label=255):
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
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss



class Dice_Loss(nn.Cell):
    def __init__(self):
        super(Dice_Loss, self).__init__()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.softmax = nn.Softmax(-1)
        self.beta = 1
        self.smooth = 1e-5
        self.Shape = ops.Shape()

    def construct(self,inputs,labels):

        n, c, h, w = list(self.Shape(inputs))
        nt, ct, ht, wt = list(self.Shape(labels))
        temp_inputs = self.transpose(self.transpose(inputs, (0, 2, 1, 3)), (0, 1, 3, 2))
        temp_inputs = self.reshape(temp_inputs,(n,-1,c))
        temp_target = self.reshape(labels,(n,-1,ct))

        # --------------------------------------------#
        #   计算dice loss
        # --------------------------------------------#
        tp = self.sum(temp_target[..., :] * temp_inputs, axis=[0, 1])
        fp = self.sum(temp_inputs, axis=[0, 1]) - tp
        fn = self.sum(temp_target[..., :], axis=[0, 1]) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        dice_loss = 1 - self.mean(score)
        return dice_loss


# 可以单独拿出来计算但是无法用于训练，只能用作监控指标

def f_score_fun(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = list(inputs.shape)
    nt, ht, wt, ct = list(target.shape)
    reshape = ops.Reshape()
    transpose = ops.Transpose()
    softmax = ops.Softmax()
    sum = ops.ReduceSum()
    mean = ops.ReduceMean()
    Resize = ops.ResizeBilinear(size=(ht,wt),align_corners=True)
    gt = ops.Greater()
    cast = ops.Cast()
    # 计算部分
    if h != ht and w != wt:
        inputs = Resize(inputs)
        inputs = cast(inputs,mindspore.float32)
    temp_inputs = transpose(transpose(inputs,(0,2,1,3)),(0,1,3,2))
    temp_inputs = reshape(temp_inputs,(n,-1,c))
    temp_inputs = softmax(temp_inputs)
    temp_target = reshape(target,(n,-1,ct))

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = gt(temp_inputs, threhold)
    tp = sum(temp_target[..., :] * temp_inputs, axis=[0, 1])
    fp = sum(temp_inputs, axis=[0, 1]) - tp
    fn = sum(temp_target[..., :], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = mean(score)
    return score

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


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)

    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                                  100 * np.nanmean(per_class_iu(hist)),
                                                                  100 * np.nanmean(per_class_PA(hist))))
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(
            round(mPA[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    return mIoUs
