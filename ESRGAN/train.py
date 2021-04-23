from time import time
import os
import argparse
import ast
import numpy as np
import cv2
import mindspore
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.ops import operations as ops
from mindspore import Tensor, context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import (
    CheckpointConfig,
    ModelCheckpoint,
    _InternalCallbackParam,
    RunContext,
)
from mindspore.ops import composite as C

from src.model.RRDB_Net import RRDBNet
from src.model.discriminator_net import VGGStyleDiscriminator128
from src.model.cell import GeneratorLossCell, DiscriminatorLossCell, TrainOneStepCellDis, TrainOneStepCellGen
from src.config.config import ESRGAN_config
from src.dataset.dataset_DIV2K import get_dataset_DIV2K


def parse_args():
    parser = argparse.ArgumentParser("ESRGAN")
    parser.add_argument('--device_target', type=str,
                        default="Ascend", help='Platform')
    parser.add_argument('--device_id', type=int,
                        default=3, help='device_id')
    parser.add_argument(
        "--aug", type=bool, default=True, help="Use augement for dataset"
    )
    parser.add_argument("--loss_scale", type=float,
                        default=1024.0, help="loss scale")
    parser.add_argument('--data_dir', type=str,
                        default=None, help='Dataset path')
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--epoch_size", type=int,
                        default=20, help="epoch_size")
    parser.add_argument('--Giters', type=int, default=5, help='number of G iters per each D iter')
    parser.add_argument("--rank", type=int, default=1,
                        help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=0, help="world size of distributed"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=30, help="max checkpoint for saving"
    )
    parser.add_argument(
        "--model_save_step", type=int, default=3000, help="step num for saving"
    )
    parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')
    parser.add_argument('--Gpretrained_path', type=str, default="src/model/psnr-1_31523.ckpt")
    parser.add_argument('--experiment', default="./images", help='Where to store samples and models')
    parser.add_argument("--run_distribute", type=ast.literal_eval,
                        default=False, help="Run distribute, default: false.")
    # Modelarts
    args, _ = parser.parse_known_args()
    return args


# save image
def save_img(img, img_name,save_dir):
    save_img = C.clip_by_value(img.squeeze(), 0, 1).asnumpy().transpose(1, 2, 0)
    # save img
    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train():
    args_opt = parse_args()
    config = ESRGAN_config
    context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", device_id=args_opt.device_id, save_graphs=False)
    # Device Environment
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            rank = args_opt.device_id
            # device_num = device_num
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        else:
            init("nccl")
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        rank = 0
        device_num = 1
    dataset, dataset_len = get_dataset_DIV2K(
        base_dir="./data", downsample_factor=config["down_factor"], mode="train", aug=args_opt.aug, repeat=1, batch_size=args_opt.batch_size,shard_id=args_opt.group_size,shard_num=args_opt.rank,num_readers=4)
    generator = RRDBNet(
        in_nc=config["ch_size"],
        out_nc=config["ch_size"],
        nf=config["G_nf"],
        nb=config["G_nb"],
    )
    discriminator = VGGStyleDiscriminator128(
        num_in_ch=config["ch_size"], num_feat=config["D_nf"])
    param_dict = load_checkpoint(args_opt.Gpretrained_path)
    load_param_into_net(generator, param_dict)
    # Define network with loss
    G_loss_cell = GeneratorLossCell(generator, discriminator,config["vgg_pretrain_path"])
    D_loss_cell = DiscriminatorLossCell(discriminator)
    lr_G = nn.piecewise_constant_lr(
        milestone=config["lr_steps"], learning_rates=config["lr_G"]
    )
    lr_D = nn.piecewise_constant_lr(
        milestone=config["lr_steps"], learning_rates=config["lr_D"]
    )
    optimizerD = nn.Adam(discriminator.trainable_params(
    ), learning_rate=lr_D, beta1=0.5, beta2=0.999,loss_scale=args_opt.loss_scale)
    optimizerG = nn.Adam(generator.trainable_params(
    ), learning_rate=lr_G, beta1=0.5, beta2=0.999,loss_scale=args_opt.loss_scale)

    # Define One step train
    G_trainOneStep = TrainOneStepCellGen(G_loss_cell, optimizerG)
    D_trainOneStep = TrainOneStepCellDis(D_loss_cell, optimizerD)

    # Train
    G_trainOneStep.set_train()
    D_trainOneStep.set_train()

    print('Start Training')

    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=args_opt.model_save_step,keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_cb_g = ModelCheckpoint(
        config=ckpt_config, directory="./checkpoints", prefix='Generator')
    ckpt_cb_d = ModelCheckpoint(
        config=ckpt_config, directory="./checkpoints", prefix='Discriminator')

    cb_params_g = _InternalCallbackParam()
    cb_params_g.train_network = generator
    cb_params_g.cur_step_num = 0
    cb_params_g.batch_num = args_opt.batch_size
    cb_params_g.cur_epoch_num = 0
    cb_params_d = _InternalCallbackParam()
    cb_params_d.train_network = discriminator
    cb_params_d.cur_step_num = 0
    cb_params_d.batch_num = args_opt.batch_size
    cb_params_d.cur_epoch_num = 0
    run_context_g = RunContext(cb_params_g)
    run_context_d = RunContext(cb_params_d)
    ckpt_cb_g.begin(run_context_g)
    ckpt_cb_d.begin(run_context_d)
    start = time()
    minibatch = args_opt.batch_size
    ones = ops.Ones()
    zeros = ops.Zeros()
    real_labels = ones((minibatch, 1), mindspore.float32)  
    fake_labels = zeros((minibatch, 1), mindspore.float32)+Tensor(np.random.random(size=(minibatch,1)),dtype=mindspore.float32)*0.1
    for iter in 
            if dis_iterations % 5 == 0:  
                save_img(target[0], 'real_samples_{0}.png'.format(dis_iterations),args_opt.experiment)
                save_img(fake_hr[0], 'fake_samples_{0}.png'.format(dis_iterations),args_opt.experiment)
if __name__ == "__main__":
    train()