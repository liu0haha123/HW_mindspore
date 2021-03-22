import mindspore
from mindspore import nn as nn
from src.archs.VGG import  vgg19

def PixelLoss(criterion="L1"):
    # 感知损失
    if criterion =="L1":
        return nn.L1Loss()
    elif criterion=="L2":
        return nn.MSELoss()

    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))


class PerceptualLoss(nn.cell):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19()
        loss_network = nn.SequentialCell(*list(vgg.features)[1:35])
        for l in loss_network.layers:
            l.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss