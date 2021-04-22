import argparse
import mindspore
import time
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import Tensor, nn
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication.management import get_rank
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore import context

from src.dataset.dataset import get_dataset_VOC, get_dataset_ADE
from src.model import PSPnet
from src.config import config
from src.utils import metrics, util
from src.utils.lr import poly_lr
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
    parser.add_argument(
        "--platform",
        type=str,
        default="Ascend",
        choices=("CPU", "GPU", "Ascend"),
        help="run platform, only support CPU, GPU and Ascend",
    )
    parser.add_argument("--device_id", type=int, default=0, help="device num")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--epoch_size", type=int,
                        default=20, help="epoch_size")
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
    parser.add_argument("--loss_scale", type=float,
                        default=1024.0, help="loss scale")
    parser.add_argument("--rank", type=int, default=2,
                        help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=1, help="world size of distributed"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="steps interval for saving"
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.0002, help="base_lr"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=10, help="max checkpoint for saving"
    )

    args = parser.parse_args()
    return args


def read_config():
    # 这里按规则返回所有的对象
    return config.pspnet_resnet50_GPU


set_seed(1)


def train():
    args_opt = parse_args()
    config = read_config()
    print(f"train args: {args_opt}\ncfg: {config}")
    
    context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", save_graphs=False, device_id=0)
    if args_opt.dataset == "ADE":

        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=config["num_classes_ADE"],
            backbone=config["backbone"],
            pretrained=True,
            pretrained_path=config["pretrained_path"],
            aux_branch=True,
        )
        dataset,dataset_len = get_dataset_ADE(
            root_path = args_opt.root_path,
            num_classes=config["num_classes_ADE"],
            mode="train",
            aug=True,
            repeat=1,
            shard_num=args_opt.rank,
            shard_id=args_opt.group_size,
            batch_size=args_opt.batch_size,
        )
        # loss
        train_net = Aux_CELoss_Cell(PSPnet_model,config["num_classes_ADE"], config["ignore_label"])
    elif args_opt.dataset == "VOC":
        PSPnet_model = PSPnet.PSPNet(
            feature_size=config["feature_size"],
            num_classes=config["num_classes"],
            backbone=config["backbone"],
            pretrained=True,
            pretrained_path=config["pretrained_path"],
            aux_branch=True,
        )
        dataset,dataset_len = get_dataset_VOC(root_path = args_opt.root_path,num_classes=config["num_classes"],
        mode="train",aug=True,repeat=1,shard_num=args_opt.rank,shard_id=args_opt.group_size,
        batch_size=args_opt.batch_size)
        train_net = Aux_CELoss_Cell(PSPnet_model,config["num_classes"], config["ignore_label"])
    else:
        raise ValueError("由于动态卷积的限制，暂不支持其他数据集")
        
    # load pretrained model
    if args_opt.ckpt_pre_trained == "train":
        param_dict = load_checkpoint(args_opt.ckpt_pre_trained)
        load_param_into_net(train_net, param_dict)
        print("load_model {} success".format(args_opt.ckpt_pre_trained))
    iters_per_epoch = dataset_len
    total_train_steps = iters_per_epoch * args_opt.epoch_size
    # get learning rate
    lr_iter = poly_lr(args_opt.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    opt = Momentum(
        params=train_net.trainable_params(),
        learning_rate=lr_iter,
        momentum=config["momentum"],
        weight_decay=0.01,
        loss_scale=args_opt.loss_scale,
    )

    # loss scale
    manager_loss_scale = FixedLossScaleManager(
        args_opt.loss_scale, drop_overflow_update=False
    )
    model = Model(train_net, optimizer=opt, amp_level="O2",loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    
    config_ck = CheckpointConfig(
        save_checkpoint_steps=args_opt.save_steps,
        keep_checkpoint_max=args_opt.keep_checkpoint_max,
    )
    ckpoint_cb = ModelCheckpoint(
        prefix=args_opt.dataset, directory="./checkpoints", config=config_ck
    )
    cbs.append(ckpoint_cb)
    model.train(
        args_opt.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=True,
    )


if __name__ == "__main__":
    train()
