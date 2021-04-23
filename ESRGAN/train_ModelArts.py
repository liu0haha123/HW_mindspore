from time import time
import os
import argparse
import ast
import numpy as np
from PIL import Image
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
import moxing as mox
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
    parser.add_argument("--data_url", type=str, default=None, help="Dataset path")
    parser.add_argument("--train_url", type=str, default=None, help="Train output path")
    parser.add_argument("--modelArts_mode", type=bool, default=True)
    # 
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
    #
    parser.add_argument("--rank", type=int, default=1,
                        help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=0, help="world size of distributed"
    )
    #
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=30, help="max checkpoint for saving"
    )
    parser.add_argument(
        "--model_save_step", type=int, default=3000, help="step num for saving"
    )
    parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')
    parser.add_argument('--experiment', default="./images", help='Where to store samples and models')
    # 
    parser.add_argument("--run_distribute", type=ast.literal_eval,
                        default=False, help="Run distribute, default: false.")
    
    args, _ = parser.parse_known_args()
    return args


def train():
    args_opt = parse_args()
    config = ESRGAN_config
    device_num = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id = int(os.getenv('RANK_ID'))
    local_data_url = "/cache/data"
    local_train_url = "/cache/lwESRGAN"
    local_zipfolder_url = "/cache/tarzip"
    local_pretrain_url = "/cache/pretrain"
    local_image_url = "/cache/ESRGANimage"
    obs_res_path = "obs://heu-535/pretrain"
    pretrain_filename = "psnr-X_XXXXX.ckpt"
    vgg_filename = ""
    filename = "DIV2K.zip"
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(local_image_url)
    context.set_context(mode=context.GRAPH_MODE,save_graphs=False,device_target="Ascend")
    # init multicards training
    if args_opt.modelArts_mode:
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        rank_id = int(os.getenv('RANK_ID'))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(device_num=device_num,parallel_mode=parallel_mode, gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()

        local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.make_dirs(local_data_url)
        local_zip_path = os.path.join(local_zipfolder_url, str(device_id), filename)
        print("device:%d, local_zip_path: %s" % (device_id, local_zip_path))
        obs_zip_path = os.path.join(args_opt.data_url, filename)
        mox.file.copy(obs_zip_path, local_zip_path)
        print(
            "====================== device %d copy end =================================\n"
            % (device_id)
        )
        unzip_command = "unzip -o %s -d %s" % (local_zip_path, local_data_url)
        os.system(unzip_command)
        print(
            "======================= device %d unzip end =================================\n"
            % (device_id)
        )
    # transfer dataset
    local_pretrain_url = os.path.join(local_zipfolder_url,pretrain_filename)
    local_pretrain_url_vgg = os.path.join(local_zipfolder_url,vgg_filename)
    obs_pretrain_url = os.path.join(obs_res_path,pretrain_filename)
    mox.file.copy(obs_pretrain_url, local_pretrain_url)
    dataset, dataset_len = get_dataset_DIV2K(base_dir=local_data_url, downsample_factor=config["down_factor"], mode="train", aug=args_opt.aug, repeat=1, batch_size=args_opt.batch_size,shard_id=args_opt.group_size,shard_num=args_opt.rank,num_readers=4)
    generator = RRDBNet(
        in_nc=config["ch_size"],
        out_nc=config["ch_size"],
        nf=config["G_nf"],
        nb=config["G_nb"],
    )
    discriminator = VGGStyleDiscriminator128(num_in_ch=config["ch_size"], num_feat=config["D_nf"])
    param_dict = load_checkpoint(local_pretrain_url)
    load_param_into_net(generator, param_dict)
    # Define network with loss
    G_loss_cell = GeneratorLossCell(generator, discriminator,local_pretrain_url_vgg)
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
        config=ckpt_config, directory=local_train_url, prefix='Generator')
    ckpt_cb_d = ModelCheckpoint(
        config=ckpt_config, directory=local_train_url, prefix='Discriminator')

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
    if device_id==0:
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
            if device_id==0:
                print('[%d/%d][%d/%d][%d] Loss_D: %10f Loss_G: %10f'
                    % (epoch, args_opt.epoch_size, i, length, dis_iterations,
                        np.sum(discriminator_loss.asnumpy()), generator_loss.asnumpy()))
                if dis_iterations % 10 == 0:  
                    save_image(target, '{0}/real_samples.png'.format(local_image_url))
                    save_image(fake_hr, '{0}/fake_samples_{1}.png'.format(local_image_url, dis_iterations))
    if device_id == 0:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
if __name__ == "__main__":
    train(config)