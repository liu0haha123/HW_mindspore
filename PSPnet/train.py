import argparse
import mindspore
import time
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import Tensor,nn
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication.management import get_rank
from mindspore.train.serialization import load_param_into_net,load_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore import context

from src.dataset.dataset import get_dataset_VOC,get_dataset_ADE
from src.model import PSPnet
from src.config import config
from src.utils import metrics,util
from src.utils.lr import get_lr

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
    parser = argparse.ArgumentParser(description='PSPnet')
    parser.add_argument('--platform', type=str, default="Ascend", choices=("CPU", "GPU", "Ascend"),
                        help='run platform, only support CPU, GPU and Ascend')
    parser.add_argument('--device_id', type=int, default=2,
                        help='device num')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--epoch_size', type=int, default=10,
                        help='epoch_size')
    parser.add_argument('--lst_path', type=str, default="./data/voc_train_lst.txt",help='Dataset List path')
    parser.add_argument('--ckpt_pre_trained', type=str, default='eval', help='train/eval')
    parser.add_argument('--loss_scale', type=float, default=3072.0, help='loss scale')
    parser.add_argument("--rank", type=int, default=2, help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=1, help="world size of distributed"
    )
    parser.add_argument(
        "--save_steps", type=int, default=3000, help="steps interval for saving"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=20, help="max checkpoint for saving"
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
    platform = args_opt.platform
    # init multicards training
    if config["run_distribute"]:
        init()
        group_size = get_group_size()

        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=2)
    PSPnet_model = PSPnet.PSPNet(feature_size=15,num_classes=21,backbone="resnet50",pretrained=True,pretrained_path=config["pretrained_path"],aux_branch=False)

    dataset = get_dataset_VOC(num_classes=21,lst_path=args_opt.lst_path,aug=True,repeat=1,shard_num=args_opt.rank,shard_id=args_opt.group_size,
                                        batch_size=8)
    # loss
    loss_ = metrics.SoftmaxCrossEntropyLoss(config["num_classes"], config["ignore_label"])
    train_net = BuildTrainNetwork(PSPnet_model, loss_)

    # load pretrained model
    if args_opt.ckpt_pre_trained == "train":
        param_dict = load_checkpoint(args_opt.ckpt_pre_trained)
        load_param_into_net(train_net, param_dict)
        print('load_model {} success'.format(args_opt.ckpt_pre_trained))
    iters_per_check = 1000
    # get learning rate
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config["lr_init"],
                       lr_end=config["lr_end"],
                       lr_max=config["lr_max"],
                       warmup_epochs=config["warmup_epochs"],
                       total_epochs=args_opt.epoch_size,
                       steps_per_epoch=iters_per_check))
    opt = Momentum(params=train_net.trainable_params(), learning_rate=lr, momentum=config["momentum"], weight_decay=0.0001,
                      loss_scale=args_opt.loss_scale)

        # loss scale
    manager_loss_scale = FixedLossScaleManager(args_opt.loss_scale, drop_overflow_update=False)
    amp_level = "O0" if args_opt.platform == "CPU" else "O3"
    model = Model(train_net, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_check)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args_opt.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_steps,
                                     keep_checkpoint_max=args_opt.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=config["name"], directory="", config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(
        args_opt.epoch_size,
        dataset,
        callbacks=cbs,
        dataset_sink_mode=False,
    )

if __name__ == "__main__":
    train()
