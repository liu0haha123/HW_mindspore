"""Evaluation"""
import os
import cv2
import time
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore import dataset as ds
from mindspore.common import set_seed
from src.model import PSPnet
from src.dataset.dataset import VOC12Dataset,ADE20k
from src.config import config
class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def parse_args(cloud_args=None):
    """parse_args"""
    parser = argparse.ArgumentParser('mindspore classification test')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    # dataset related
    parser.add_argument('--data_path', type=str,
                        default='./data', help='eval data dir')
    parser.add_argument(
        "--dataset",
        type=str,
        default="VOC",
        help="ADE / VOC",
    )
    # val data
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--crop_size', type=int, default=473, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--ckpt_path', type=str,
                        default='./checkpoints/ADE_2-12_631.ckpt', help='eval data dir')
    parser.add_argument('--scales', type=float, action='append',default=1.0,help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', default=True,help='perform left-right flip')
    parser.add_argument('--crop_size', type=int, default=473, help='crop size')
    args_opt = parser.parse_args()
    return args_opt


def read_config():
    # 这里按规则返回所有的对象
    return config.pspnet_resnet50_GPU

def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=473):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(args, img_, crop_size=473):
    """pre_process"""
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(args, eval_net, img_lst, crop_size=473, flip=True):
    """eval_batch"""
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=473, flip=True):
    """eval_batch_scales"""
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


set_seed(1)


def test():
    args_opt = parse_args()
    config = read_config()
    print(f"test args: {args_opt}\ncfg: {config}")
    context.set_context(
        mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=2
    )
    if args_opt.dataset == "VOC":
        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=args_opt.num_classes,
            backbone=config["backbone"],
            pretrained=False,
            pretrained_path=config["pretrained_path"],
            aux_branch=False,
        )

        dataset_eval = VOC12Dataset(root_path=args_opt.data_path,num_classes=args_opt.num_classes,aug=False,mode="eval")
        dataset_eval = ds.GeneratorDataset(source=dataset_eval, column_names=["data", "label"],
                                       shuffle=False)
        dataset_eval = dataset_eval.batch(1)
    else :
        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=config["num_classes_ADE"],
            backbone=config["backbone"],
            pretrained=True,
            pretrained_path=config["pretrained_path"],
            aux_branch=True,
        )
        dataset_eval = ADE20k(root_path=args_opt.data_path,num_classes=args_opt.num_classes,aug=False,mode="eval")
        dataset_eval = ds.GeneratorDataset(source=dataset_eval, column_names=["data", "label"],
                                       shuffle=False)
        dataset_eval = dataset_eval.batch(1)
    eval_net = BuildEvalNetwork(network=PSPnet_model)
    # load model
    param_dict = load_checkpoint(args_opt.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)
    test_data_iter = dataset_eval.create_dict_iter()
    # evaluate
    hist = np.zeros((args_opt.num_classes, args_opt.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    for i, sample in enumerate(test_data_iter):
        image = sample['data'].asnumpy()
        label = sample['label'].asnumpy()
        batch_img_lst.append(image)
        batch_msk_lst.append(label)
        bi += 1
        if bi == args_opt.batch_size:
            batch_res = eval_batch_scales(args_opt, eval_net, batch_img_lst, scales=args_opt.scales,
                                          base_crop_size=args_opt.crop_size, flip=args_opt.flip)
            for mi in range(args_opt.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args_opt.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            print('processed {} images'.format(i + 1))
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args_opt, eval_net, batch_img_lst, scales=args_opt.scales,
                                      base_crop_size=args_opt.crop_size, flip=args_opt.flip)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args_opt.num_classes)
        print('processed {} images'.format(image_num + 1))

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))

    """
    pred_label_for_matrix = []
    pred_label_for_matrix = []

    # 指标初始化
    train_pa = 0
    train_miou = 0
    train_mpa = 0
    error = 0
    for i, sample in enumerate(test_data):
        image = sample['data']
        label = sample['label']

        out = eval_net(image)
        pred_label = ops.ReduceMax(dim=1)(out)[1].asnumpy()
        for p in pred_label:
            pred_label_for_matrix.append(p)

        true_label = label.asnumpy()
        for t in true_label:
            pred_label_for_matrix.append(t)
        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        test_pa = eval_metrix['pa'] + train_pa
        test_miou = eval_metrix['miou'] + train_miou
        test_mpa = eval_metrix["mpa"] + train_mpa

        if i % 10 == 0:
            # 定义打印格式
            epoch_str = ('test_pa :{:.5f} , test_miou:{:.5f}'.format(
                train_pa / len(test_data), train_miou/len(test_data)))
            print(epoch_str)
    final_pa = test_pa/len(dataset)
    final_miou = test_miou/len(dataset)
    final_str = ('final_pa :{:.5f} , final_miou:{:.5f}'.format(
        final_pa / len(test_data), final_miou/len(test_data)))
     """
if __name__ == "__main__":
    test()
