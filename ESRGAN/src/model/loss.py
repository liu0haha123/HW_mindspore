import mindspore
from mindspore import nn as nn
from src.model.VGG import vgg19
import mindspore.ops.functional as F
from mindspore.train.serialization import load_checkpoint, load_param_into_net
class PerceptualLoss(nn.Cell):
    # 内容损失
    def __init__(self,pretrained_path):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19()
        loss_network = vgg.layers[:35]
        param_dict = load_checkpoint(pretrained_path)
        load_param_into_net(vgg,param_dict)
        for l in loss_network:
            l.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def construct(self, high_resolution, fake_high_resolution):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 读取时放缩到了0-1 在计算内容损失时应该还原
        # 12.75 is rescale factor for vgg featuremaps.
        perception_loss = self.l1_loss(
            (self.loss_network(high_resolution * 255.0) / 12.75),
            (self.loss_network(fake_high_resolution * 255.0) / 12.75),
        )
        return perception_loss

