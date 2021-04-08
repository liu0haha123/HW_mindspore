import argparse
import mindspore
import os
import time
import random
import numpy as np
from mindspore import Tensor,DatasetHelper,nn
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed

from src.dataset import dataset
from src.model import PSPnet
from src.config import config
from src.utils import lr,metrics,util
# 定义训练/验证时指定的可变参数

def parse_args():
    parser = argparse.ArgumentParser(description='PSPnet')
    parser.add_argument('--platform', type=str, default="GPU", choices=("CPU", "GPU", "Ascend"),
                        help='run platform, only support CPU, GPU and Ascend')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device num')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--epoch_size', type=int, default=10,
                        help='epoch_size')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    parser.add_argument('--mode', type=str, default='train', help='train/eval')

    args = parser.parse_args()
    return args

def read_config():
    # 这里按规则返回所有的对象
    return config.pspnet_resnet50_GPU

set_seed(1)

if __name__ == '__main__':
    args_opt = parse_args()
    config = read_config()
    start = time.time()
    print(f"train args: {args_opt}\ncfg: {config}")
    rank_size = mindspore.communication.get_local_rank_size()
    device_id = args_opt.device_id
    platform = args_opt.platform
    #set context and device init
    util.context_device_init(platform,config.run_distribute,device_id,rank_size)
    PSPnet_model = PSPnet.PSPNet(input_size=config.input_size[0:1],feature_size=15,num_classes=21,backbone="resnet50",pretrained=True,pretrained_path="",aux_branch=True)

    dataset_train = dataset.create_dataset("VOC2012",dataset_path=args_opt.dataset_path,mode=args_opt.mode,platform=platform,run_distribute=config.run_distribute,
                                           resize_shape=config.resize_shape,batch_size=args_opt.batch_size,repeat_num=1)
    # Currently, only Ascend support switch precision.
    util.switch_precision(PSPnet_model, mstype.float16, platform)
    epoch_size = args_opt.epoch_size
    # define loss
    loss = mindspore.nn.DiceLoss(smooth=1e-5)
    # get learning rate
    lr = Tensor(lr.get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size,
                       steps_per_epoch=len(dataset_train)))
    # define optimizer
    opt = Momentum(filter(lambda x: x.requires_grad, PSPnet_model.get_parameters()), lr, config.momentum,
                   config.weight_decay)
    #构建网络
    dataset_helper = DatasetHelper(dataset_train,sink_size=100,epoch_num=epoch_size)
    network = WithLossCell(PSPnet_model, loss)
    network = TrainOneStepCell(network, opt)
    network.set_train()
    rank = 0
    if config.run_distribute:
        rank = get_rank()
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
    if not os.path.isdir(save_ckpt_path):
        os.mkdir(save_ckpt_path)

    for epoch in range(epoch_size):
        epoch_start = time.time()
        losses = []
        for data in dataset_helper:
            inputs,label,seg_label = data[0],data[1],data[2]
            loss = network(inputs,label)
            losses.append(loss.asnumpy())
        epoch_mseconds = (time.time()-epoch_start) * 1000
        per_step_mseconds = epoch_mseconds / len(dataset_train)
        print("epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}"\
        .format(epoch + 1, epoch_size, len(dataset_train), epoch_mseconds, per_step_mseconds, np.mean(np.array(losses))))
        if (epoch + 1) % config.save_checkpoint_epochs == 0:
            save_checkpoint(PSPnet_model, os.path.join(save_ckpt_path, f"PSPnet_{epoch+1}.ckpt"))
    print("total cost {:5.4f} s".format(time.time() - start))


