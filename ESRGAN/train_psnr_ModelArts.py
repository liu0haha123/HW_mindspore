import mindspore
import moxing as mox
import os
from mindspore import nn
from src.dataset.dataset_DIV2K import get_dataset_DIV2K
from src.model.RRDB_Net import RRDBNet
from src.config import config
from mindspore.parallel import set_algo_parameters
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import context
import argparse
from mindspore.communication.management import init


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def parse_args():
    parser = argparse.ArgumentParser("Generator Pretrain")
    parser.add_argument("--data_url", type=str, default=None, help="Dataset path")
    parser.add_argument("--train_url", type=str, default=None, help="Train output path")
    parser.add_argument("--modelArts_mode", type=bool, default=True)

    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="device id of GPU or Ascend. (Default: None)",
    )
    parser.add_argument(
        "--aug", type=bool, default=True, help="Use augement for dataset"
    )
    parser.add_argument("--loss_scale", type=float,
                        default=1024.0, help="loss scale")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--epoch_size", type=int, default=20, help="epoch_size")
    parser.add_argument("--rank", type=int, default=0, help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=1, help="world size of distributed"
    )
    parser.add_argument(
        "--save_steps", type=int, default=2000, help="steps interval for saving"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=20, help="max checkpoint for saving"
    )
    # 分布式

    parser.add_argument("--distribute", type=bool, default=True, help="run distribute")

    args, _ = parser.parse_known_args()
    return args


def train(config):
    args_opt = parse_args()
    config_psnr = config.PSNR_config
    # 这里开始 ModelArts部分
    device_num = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id = int(os.getenv('RANK_ID'))
    local_data_url = "/cache/data"
    local_train_url = "/cache/lwESRGAN"
    local_zipfolder_url = "/cache/tarzip"
    local_pretrain_url = "/cache/pretrain"
    obs_res_path = "obs://heu-535/pretrain"
    pretrain_filename = "vgg19_ImageNet.ckpt"
    filename = "DIV2K.zip"
    mox.file.make_dirs(local_train_url)
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
    obs_pretrain_url = os.path.join(obs_res_path,pretrain_filename)
    mox.file.copy(obs_pretrain_url, local_pretrain_url)

    model_psnr = RRDBNet(
        in_nc=config_psnr["ch_size"],
        out_nc=config_psnr["ch_size"],
        nf=config_psnr["G_nf"],
        nb=config_psnr["G_nb"],
    )
    dataset,dataset_len = get_dataset_DIV2K(
        base_dir=local_data_url,
        downsample_factor=config_psnr["down_factor"],
        mode="train",
        aug=args_opt.aug,
        repeat=1,
        num_readers=4,
        shard_id=args_opt.rank,
        shard_num=args_opt.group_size,
        batch_size=args_opt.batch_size,
    )

    lr = nn.piecewise_constant_lr(
        milestone=config_psnr["lr_steps"], learning_rates=config_psnr["lr"]
    )
    opt = nn.Adam(
        params=model_psnr.trainable_params(), learning_rate=lr, beta1=0.9, beta2=0.99,loss_scale=args_opt.loss_scale
    )
    loss = nn.L1Loss()
    loss.add_flags_recursive(fp32=True)
    amp_level = "O2"
    train_net = BuildTrainNetwork(model_psnr, loss)
    iters_per_check = dataset_len
    model = Model(train_net, optimizer=opt)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_check)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]


    config_ck = CheckpointConfig(
        save_checkpoint_steps=args_opt.save_steps,
        keep_checkpoint_max=args_opt.keep_checkpoint_max,
    )
    ckpoint_cb = ModelCheckpoint(
        prefix="psnr", directory=local_train_url, config=config_ck
    )
    if device_id ==0:
        cbs.append(ckpoint_cb)

    model.train(
        args_opt.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=True,
    )

    if device_id == 0:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)


if __name__ == "__main__":
    train(config)
