import argparse
import mindspore
import time
import os
from mindspore.parallel import set_algo_parameters
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import Tensor, nn
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore import context
import moxing as mox
from src.dataset.dataset import get_dataset_VOC, get_dataset_ADE
from src.model import PSPnet
from src.config import config
from src.utils import lr
from src.model.cell import Aux_CELoss_Cell


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
    parser = argparse.ArgumentParser(description="PSPnet")

    parser.add_argument("--data_url", type=str, default=None, help="Dataset path")
    parser.add_argument("--train_url", type=str, default=None, help="Train output path")
    parser.add_argument("--modelArts_mode", type=bool, default=True)

    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        choices=("CPU", "GPU", "Ascend"),
        help="run platform, only support CPU, GPU and Ascend",
    )
    parser.add_argument("--device_id", type=int, default=2, help="device num")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--base_lr", type=int, default=0.01, help="base_lr")
    parser.add_argument("--epoch_size", type=int,
                        default=50, help="epoch_size")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data",
        help="Dataset List path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ADE",
        help="ADE / VOC",
    )
    parser.add_argument(
        "--ckpt_pre_trained", type=str, default="eval", help="train/eval"
    )
    parser.add_argument(
        "--pretrained_path", type=str, help="pretrain resnet"
    )
    parser.add_argument("--loss_scale", type=float,
                        default=1024.0, help="loss scale")
    parser.add_argument("--rank", type=int, default=0,
                        help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=1, help="world size of distributed"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="steps interval for saving"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=20, help="max checkpoint for saving"
    )
    # 分布式

    parser.add_argument("--distribute", type=bool, default=False, help="run distribute")

    args = parser.parse_args()
    return args


def read_config():
    # 这里按规则返回所有的对象
    return config.pspnet_resnet50_GPU


set_seed(1)


def train():
    args_opt = parse_args()
    config = read_config()
    # 这里开始
    device_num = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id = int(os.getenv('RANK_ID'))
    args_opt.group_size = device_num
    local_data_url = "/cache/data"
    local_train_url = "/cache/lwPSP"
    local_zipfolder_url = "/cache/tarzip"
    local_pretrain_url = "/cache/pretrain"
    obs_res_path = "obs://heu-535/pretrain"
    pretrain_filename = "resnet.ckpt"
    filename = "data_PSP.zip"
    mox.file.make_dirs(local_train_url)
    print(f"train args: {args_opt}\ncfg: {config}")
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend")
    # init multicards training
    if args_opt.modelArts_mode:
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        rank_id = int(os.getenv('RANK_ID'))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=parallel_mode, gradients_mean=True)
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
    local_pretrain_url = os.path.join(local_pretrain_url, pretrain_filename)
    obs_pretrain_url = os.path.join(obs_res_path, pretrain_filename)
    mox.file.copy(obs_pretrain_url, local_pretrain_url)
    local_data_url = local_data_url + "/data"

    if args_opt.dataset == "ADE":

        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=config["num_classes_ADE"],
            backbone=config["backbone"],
            pretrained=True,
            pretrained_path=local_pretrain_url,
            aux_branch=True,
        )
        dataset, dataset_len = get_dataset_ADE(
            root_path=local_data_url,
            num_classes=config["num_classes_ADE"],
            mode="train",
            aug=True,
            repeat=1,
            shard_num=args_opt.group_size,
            shard_id=args_opt.rank,
            batch_size=args_opt.batch_size,
            num_workers=8
        )
        # loss
        train_net = Aux_CELoss_Cell(PSPnet_model, config["num_classes_ADE"], config["ignore_label"])
    elif args_opt.dataset == "VOC":
        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=config["num_classes"],
            backbone=config["backbone"],
            pretrained=True,
            pretrained_path=local_pretrain_url,
            aux_branch=True,
        )
        dataset, dataset_len = get_dataset_VOC(root_path=local_data_url, num_classes=config["num_classes"],
                                               mode="train", aug=True, repeat=1, shard_num=args_opt.group_size,
                                               shard_id=args_opt.rank,
                                               batch_size=args_opt.batch_size, num_workers=8)
        train_net = Aux_CELoss_Cell(PSPnet_model, config["num_classes"], config["ignore_label"])
    else:
        raise ValueError("由于动态卷积的限制，暂不支持其他数据集")
    train_net.to_float(mindspore.float32)
    # load pretrained model
    if args_opt.ckpt_pre_trained == "train":
        param_dict = load_checkpoint(args_opt.ckpt_pre_trained)
        load_param_into_net(train_net, param_dict)
        print("load_model {} success".format(args_opt.ckpt_pre_trained))
    iters_per_epoch = dataset_len
    total_train_steps = iters_per_epoch * args_opt.epoch_size
    # get learning rate
    lr_iter = lr.poly_lr(args_opt.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    opt = Momentum(
        params=train_net.trainable_params(),
        learning_rate=lr_iter,
        momentum=config["momentum"],
        weight_decay=0.0001,
        loss_scale=args_opt.loss_scale,
    )

    # loss scale
    manager_loss_scale = FixedLossScaleManager(
        args_opt.loss_scale, drop_overflow_update=False
    )
    amp_level = "O2"
    model = Model(train_net, optimizer=opt, amp_level=amp_level)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    config_ck = CheckpointConfig(
        save_checkpoint_steps=args_opt.save_steps,
        keep_checkpoint_max=args_opt.keep_checkpoint_max,
    )
    ckpoint_cb = ModelCheckpoint(
        prefix=args_opt.dataset, directory=local_train_url, config=config_ck
    )
    if device_id == 0:
        cbs.append(ckpoint_cb)
    model.train(
        args_opt.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=True,
    )
    if device_id == 0:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)


if __name__ == "__main__":
    train()
