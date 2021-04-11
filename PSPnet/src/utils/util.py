import time
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore.communication.management import get_rank, init, get_group_size
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """
    incremental_learning or not
    """
    param_dict = load_checkpoint(pretrain_ckpt_path)
    load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False



def switch_precision(net, data_type, platform):
    if platform == "Ascend":
        net.to_float(data_type)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)

def context_device_init(platform,run_distribute,device_id,rank_size):
    if platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=platform, save_graphs=False)

    elif platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=platform, save_graphs=False)
        if run_distribute:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    elif platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=platform, device_id=device_id,
                            save_graphs=False)
        if run_distribute:
            context.set_auto_parallel_context(device_num=rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


def set_context(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            save_graphs=False)
    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            device_id=config.device_id, save_graphs=False)
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=config.platform, save_graphs=False)

