import mindspore
from mindspore import nn
from src.dataset.dataset_DIV2K import get_dataset_DIV2K
from src.model.RRDB_Net import RRDBNet
from src.config import config
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
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="device id of GPU or Ascend. (Default: None)",
    )
    parser.add_argument(
        "--aug", type=bool, default=True, help="Use augement for dataset"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--epoch_size", type=int, default=20, help="epoch_size")
    parser.add_argument("--rank", type=int, default=0, help="local rank of distributed")
    parser.add_argument(
        "--group_size", type=int, default=1, help="world size of distributed"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="steps interval for saving"
    )
    parser.add_argument(
        "--keep_checkpoint_max", type=int, default=20, help="max checkpoint for saving"
    )
    args, _ = parser.parse_known_args()
    return args


def train(config):

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2)
    args = parse_args()
    config_psnr = config.PSNR_config
    model_psnr = RRDBNet(
        in_nc=config_psnr["ch_size"],
        out_nc=config_psnr["ch_size"],
        nf=config_psnr["G_nf"],
        nb=config_psnr["G_nb"],
    )
    dataset,dataset_len = get_dataset_DIV2K(
        base_dir="./data",
        downsample_factor=config_psnr["down_factor"],
        mode="train",
        aug=args.aug,
        repeat=1,
        num_readers=4,
        shard_id=args.rank,
        shard_num=args.group_size,
        batch_size=args.batch_size,
    )

    lr = nn.piecewise_constant_lr(
        milestone=config_psnr["lr_steps"], learning_rates=config_psnr["lr"]
    )
    opt = nn.Adam(
        params=model_psnr.trainable_params(), learning_rate=lr, beta1=0.9, beta2=0.99
    )
    loss = nn.L1Loss()
    loss.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(model_psnr, loss)

    model = Model(train_net, optimizer=opt)
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=1000)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=args.save_steps,
            keep_checkpoint_max=args.keep_checkpoint_max,
        )
        ckpoint_cb = ModelCheckpoint(
            prefix="psnr", directory="./checkpoints", config=config_ck
        )
        cbs.append(ckpoint_cb)

    model.train(
        args.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=False,
    )


if __name__ == "__main__":
    train(config)
