"""Evaluation"""
import os
import time
import argparse
import datetime
import glob
import numpy as np
import cv2
import mindspore.nn as nn
from mindspore.nn import PSNR,SSIM
from mindspore import Tensor, context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from src.config.config import ESRGAN_config,PSNR_config
from src.utils.eval_util import imresize_np, rgb2ycbcr, calculate_psnr, calculate_ssim


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network

    def construct(self, input_data):
        output = self.network(input_data)
        return output


def parse_args(cloud_args=None):
    """parse_args"""
    parser = argparse.ArgumentParser('Eval ESRGAN')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    # dataset related
    parser.add_argument('--data_path', type=str,
                        default='', help='eval data dir')
    parser.add_argument('--batch_size', default=1,
                        type=int, help='batch size for per npu')
    # network related
    parser.add_argument('--graph_ckpt', type=int, default=1,
                        help='graph ckpt or feed ckpt')
    parser.add_argument('--pre_trained', default='', type=str, help='fully path of pretrained model to load. '
                        'If it is a direction, it will test all ckpt')

    # logging related
    parser.add_argument('--log_path', type=str,
                        default='outputs/', help='path to save log')
    parser.add_argument('--rank', type=int, default=0,
                        help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1,
                        help='world size of distributed')

    args_opt = parser.parse_args()
    return args_opt

set_seed(1)


def test():
    args_opt = parse_args()
    config = PSNR_config
    print(f"test args: {args_opt}\ncfg: {config}")
    context.set_context(
        mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=1
    )

    model_psnr = RRDBNet(
        in_nc=config_psnr["ch_size"],
        out_nc=config_psnr["ch_size"],
        nf=config_psnr["G_nf"],
        nb=config_psnr["G_nb"],
    )
    dataset,dataset_len = get_dataset_DIV2K(
        base_dir="./data",
        downsample_factor=config_psnr["down_factor"],
        mode="valid",
        aug=False,
        repeat=1,
        num_readers=4,
        shard_id=args.rank,
        shard_num=args.group_size,
        batch_size=1,
    )

    eval_net = BuildEvalNetwork(model_psnr)

    # load model
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)
    ssim = nn.SSIM()
    psnr = nn.PSNR()
    test_data_iter = dataset.create_dict_iter(out_numpy=False)
    psnr_bic_all = 0.0
    psnr_real_all = 0.0
    ssim_bic_all = 0.0
    ssim_real_all = 0.0
    for i, sample in enumerate(test_data):
        lr = sample['inputs']
        real_hr = sample['target']
        gen_hr = eval_net(lr)
        # 这里用mindspore的双三次插值采样结果  
        bic_hr = None
        psnr_bic = psnr(gen_hr,bic_hr)
        psnr_real = psnr(gen_hr,real_hr)
        ssim_bic = ssim(gen_hr,bic_hr)
        ssim_real = ssim(gen_hr,real_hr)
        psnr_bic_all += psnr_bic
        psnr_real_all = psnr_real
        ssim_bic_all = ssim_bic
        ssim_real_all = ssim_real
        print(psnr_bic,psnr_real,ssim_bic,ssim_real)
        result_img_path = os.path.join(args_opt.results_path + "DIV2K", 'Bic_SR_HR_' + str(i))
        if i%10 == 0:
            results_img = np.concatenate((bic_img[0].asnumpy(), sr_img[0].asnumpy(), hr_img[0].asnumpy()), 1)
            cv2.imwrite(result_img_path, results_img)
    psnr_bic_all += psnr_bic_all/dataset_len
    psnr_real_all = psnr_real_all/dataset_len
    ssim_bic_all = ssim_bic_all/dataset_len
    ssim_real_all = ssim_real_all/dataset_len
    print(psnr_bic_all,psnr_real_all,ssim_bic_all,ssim_real_all)