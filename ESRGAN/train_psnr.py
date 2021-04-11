import mindspore
from mindspore import nn
from src.dataset.dataset_DIV2K import get_dataset_DIV2K
from src.model.RRDB_Net import RRDBNet
from src.config import config
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss

def train():
    config_psnr = config.PSNR_config
    model_psnr = RRDBNet(in_nc=3,out_nc=64,nf=config_psnr["G_nf"],nb=config_psnr["G_nb"])
    dataset = get_dataset_DIV2K(base_dir="",downsample_factor=4,mode="train",aug=True,repeat=1,num_readers=4,shard_id=0,shard_num=1,batch_size=8)

    lr = nn.piecewise_constant_lr(milestone=config_psnr["lr_steps"],learning_rates=config_psnr["lr"])
    opt = nn.Adam(params=model_psnr.trainable_params(),learning_rate=lr,beta1=0.9,beta2=0.99)
    loss = nn.L1Loss()
    loss.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(model_psnr, loss)

    model = Model(train_net, optimizer=opt)
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=1000)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    model.train(
        2000,
        dataset,
        callbacks=cbs,
        dataset_sink_mode=False,
    )