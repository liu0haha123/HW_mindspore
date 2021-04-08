import mindspore
from mindspore import nn as nn
from src.model.VGG import vgg19
import mindspore.ops.functional as F

class PixelLoss(nn.Cell):
    # 感知损失
    def __init__(self,criterion='l1'):
        super(PixelLoss, self).__init__()
        if criterion == 'l1':
            self.loss = nn.L1Loss()
        elif criterion == 'l2':
            self.loss = nn.MSELoss()

    def construct(self, hr,sr):
        return self.loss(hr,sr)


class PerceptualLoss(nn.cell):
    # 内容损失
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19()
        loss_network = nn.SequentialCell(*list(vgg.features)[1:35])
        for l in loss_network.layers:
            l.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def construct(self, high_resolution, fake_high_resolution):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        perception_loss = self.l1_loss((self.loss_network(high_resolution* 255.)/12.75), (self.loss_network(fake_high_resolution* 255.)/12.75))
        return perception_loss



def DiscriminatorLoss(gan_type='ragan'):
    """discriminator loss"""
    binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
    cross_entropy = binary_cross_entropy
    sigma = mindspore.ops.Sigmoid
    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(F.ones_like(hr), sigma(hr - mindspore.ops.ReduceMean(sr))) +
            cross_entropy(F.zeros_like(sr), sigma(sr -  mindspore.ops.ReduceMean(hr))))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(F.ones_like(hr), sigma(hr))
        fake_loss = cross_entropy(F.zeros_like(sr), sigma(sr))
        return real_loss + fake_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))

def GeneratorLoss(gan_type='ragan'):
    """generator loss"""
    binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
    cross_entropy = binary_cross_entropy
    sigma = mindspore.ops.Sigmoid

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(F.ones_like(sr), sigma(sr - mindspore.ops.ReduceMean(hr))) +
            cross_entropy(F.zeros_like(sr), sigma(hr -  mindspore.ops.ReduceMean(sr))))
    def generator_loss(hr, sr):
        return cross_entropy(F.ones_like(sr), sigma(sr))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
