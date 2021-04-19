from time import time
import os
import argparse
import ast
import numpy as np
import mindspore
import mindspore.common.dtype as mstype
from mindspore import nn
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

from src.model.RRDB_Net import RRDBNet
from src.model.discriminator_net import VGGStyleDiscriminator512
from src.model.cell import GeneratorLossCell, DiscriminatorLossCell, TrainOneStepCellDis, TrainOneStepCellGen
from src.config.config import ESRGAN_config
from src.dataset.dataset_DIV2K import get_dataset_DIV2K


def parse_args():
    parser = argparse.ArgumentParser("ESRGAN")
    parser.add_argument('--device_target', type=str,
                        default="Ascend", help='Platform')
    parser.add_argument('--device_id', type=int,
                        default=, help='device_id')
    parser.add_argument(
        "--aug", type=bool, default=True, help="Use augement for dataset"
    )
    parser.add_argument('--data_dir', type=str,
                        default=None, help='Dataset path')
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--epoch_size", type=int,
                        default=20, help="epoch_size")
    parser.add_argument("--rank", type=int, default=1,
                        help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=0, help="world size of distributed"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=20, help="max checkpoint for saving"
    )
    parser.add_argument(
        "--model_save_step", type=int, default=3000, help="step num for saving"
    )
    parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')
    parser.add_argument("--run_distribute", type=ast.literal_eval,
                        default=False, help="Run distribute, default: false.")
    # Modelarts
    args, _ = parser.parse_known_args()
    return args


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
    discriminator = VGGStyleDiscriminator512(
        num_in_ch=config["ch_size"], num_feat=config["D_nf"])
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
    ), learning_rate=lr_D, beta1=0.5, beta2=0.999)
    optimizerG = nn.Adam(generator.trainable_params(
    ), learning_rate=lr_G, beta1=0.5, beta2=0.999)

    # Define One step train
    G_trainOneStep = TrainOneStepCellGen(G_loss_cell, optimizerG)
    D_trainOneStep = TrainOneStepCellDis(D_loss_cell, optimizerD)

    # Train
    G_trainOneStep.set_train()
    D_trainOneStep.set_train()

    print('Start Training')

    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=args_opt.model_save_step)
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
    mean = ops.ReduceMean(keep_dims=True)
    for epoch in range(args_opt.epoch_size):
        G_epoch_loss = 0
        D_epoch_loss = 0
        G_content_epoch_loss = 0
        G_perception_epoch_loss = 0
        G_adversarial_epoch_loss = 0

        for iteration, batch in enumerate(dataset.create_dict_iterator(), 1):
            inputs = Tensor(batch["inputs"],dtype=mindspore.float32)
            target = Tensor(batch["target"],dtype=mindspore.float32)
            minibatch = inputs.shape[0]
            ones = ops.Ones()
            zeros = ops.Zeros()
            real_labels = ones((minibatch, 1), mindspore.float32)  
            fake_labels = zeros((minibatch, 1), mindspore.float32)  # torch.rand(minibatch,1)*0.3
            generator_loss_all = G_trainOneStep(
                inputs, target, fake_labels, real_labels)
            fake_hr = generator_loss_all[0]
            generator_loss = generator_loss_all[1]
            discriminator_loss = D_trainOneStep(fake_hr,target)
            G_epoch_loss += generator_loss.asnumpy()
            D_epoch_loss += np.sum(discriminator_loss.asnumpy())
            print(epoch,iteration, dataset_len, generator_loss.asnumpy(), np.sum(discriminator_loss.asnumpy()))

        print(
        "===> Epoch: [%5d] Complete: Avg. Loss G: %.4f D: %.4f" %(
            epoch, np.true_divide(G_epoch_loss, dataset_len), np.true_divide(D_epoch_loss, dataset_len)))
        if (epoch+1) % (opt.snapshots) == 0:
            print('===> Saving model')
            cb_params_d.cur_step_num = epoch + 1
            cb_params_g.cur_step_num = epoch + 1
            ckpt_cb_g.step_end(run_context_g)
            ckpt_cb_d.step_end(run_context_d)


if __name__ == '__main__':
    train()     
