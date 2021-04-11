# @author AmythistHe
# @version 1.0
# @description
# @create 2021/3/22 21:17
import argparse
import mindspore
import os
from mindspore import context, Tensor
import numpy as np
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore import context
import mindspore.dataset as ds
import pdb
import socket
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/hoo/ms_dataset/DIV2K')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')

opt = parser.parse_args()
# gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)

# 调用集合通信库
"""
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
init()
"""

# 创建数据集
print('===> Loading datasets')
# 分布式
"""
rank_id = get_rank()  # 获取当前设备在集群中的ID
rank_size = get_group_size()  # 获取集群数量
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
                                           num_parallel_workers=opt.threads, shuffle=True,
                                           num_shards=rank_size, shard_id=rank_id)
"""
# 单卡
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
                                           num_parallel_workers=opt.threads, shuffle=True)



# 模型载入
print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64, feat=256, num_stages=3, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=opt.upscale_factor)


# 模型训练


# 模型测试
test_Input = Tensor(np.random.rand(1, 3, 40, 40), dtype=mindspore.float32)
test_Target = Tensor(np.random.rand(1, 3, 320, 320), dtype=mindspore.float32)
test_Bicubic = Tensor(np.random.rand(1, 3, 320, 320), dtype=mindspore.float32)

prediction = model(test_Input)
prediction = prediction + test_Bicubic
print(prediction.shape)


