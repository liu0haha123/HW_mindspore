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
from src.model.discriminator_net import VGGStyleDiscriminator128
from src.model.cell import GeneratorLossCell, DiscriminatorLossCell, TrainOneStepCellDis, TrainOneStepCellGen
from src.config.config import ESRGAN_config
from src.dataset.dataset_DIV2K import get_dataset_DIV2K

# save image
def save_image(img, img_path):
    mul = ops.Mul()
    add = ops.Add()
    if isinstance(img, Tensor):
        img = mul(img, 0.5)
        img = add(img, 0.5)
        img = img.asnumpy().astype(np.uint8).transpose((0, 2, 3, 1))

    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

    IMAGE_SIZE = 64  # Image size
    IMAGE_ROW = 8  # Row num
    IMAGE_COLUMN = 8  # Column num
    PADDING = 2 #Interval of small pictures
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + PADDING * (IMAGE_COLUMN + 1),
                                    IMAGE_ROW * IMAGE_SIZE + PADDING * (IMAGE_ROW + 1)))  # create a new picture
    # cycle
    i = 0
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.fromarray(img[i])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE + PADDING * x, (y - 1) * IMAGE_SIZE + PADDING * y))
            i = i + 1

    to_image.save(img_path)  #save


def parse_args():
    parser = argparse.ArgumentParser("ESRGAN")
    parser.add_argument('--device_target', type=str,
                        default="Ascend", help='Platform')
    parser.add_argument('--device_id', type=int,
                        default=1, help='device_id')
    parser.add_argument(
        "--aug", type=bool, default=True, help="Use augement for dataset"
    )
    parser.add_argument('--data_dir', type=str,
                        default=None, help='Dataset path')
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
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
    parser.add_argument('--experiment', default="./images", help='Where to store samples and models')
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
    discriminator = VGGStyleDiscriminator128(
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
    dis_iterations = 0
    for epoch in range(args_opt.epoch_size):
        data_iter = dataset.create_dict_iterator()
        length = dataset_len
        i = 0
        while i < length:
            ############################
            # (1) Update G network
            ###########################
            for p in generator.trainable_params():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if dis_iterations < 25 or dis_iterations % 500 == 0:
                Giters = 100
            else:
                Giters = args_opt.Giters
            j = 0
            while j < Giters and i < length:
                j += 1

                # clamp parameters to a cube
                # for p in netD.trainable_params():
                #    p.data.clamp_(args_opt.clamp_lower, args_opt.clamp_upper)

                data = data_iter.__next__()
                i += 1

                # train with real and fake
                inputs = Tensor(data["inputs"],dtype=mindspore.float32)
                target = Tensor(data["target"],dtype=mindspore.float32)
                generator_loss_all = G_trainOneStep(inputs, target, fake_labels, real_labels)
                fake_hr = generator_loss_all[0]
                generator_loss = generator_loss_all[1]

            ############################
            # (2) Update G network
            ###########################
            for p in generator.trainable_params():
                p.requires_grad = False  # to avoid computation

            discriminator_loss = D_trainOneStep(fake_hr,target)
            dis_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f'
                  % (epoch, args_opt.epoch_size, i, length, dis_iterations,
                     np.sum(discriminator_loss.asnumpy()), generator_loss.asnumpy()))
            if dis_iterations % 10 == 0:  
                save_image(real, '{0}/real_samples.png'.format(args_opt.experiment))
                save_image(loss_G[0], '{0}/fake_samples_{1}.png'.format(args_opt.experiment, gen_iterations))

train()