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

def DiscriminatorLoss(gan_type='ragan'):
    """discriminator loss"""
    binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
    cross_entropy = binary_cross_entropy
    sigma = mindspore.ops.Sigmoid
    oneslike = mindspore.ops.OnesLike()
    zeroslike = mindspore.ops.ZerosLike()
    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(oneslike(hr), sigma(hr - mindspore.ops.ReduceMean(sr))) +
            cross_entropy(zeroslike(sr), sigma(sr -  mindspore.ops.ReduceMean(hr))))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(oneslike(hr), sigma(hr))
        fake_loss = cross_entropy(zeroslike(sr), sigma(sr))
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
    oneslike = mindspore.ops.OnesLike()
    zeroslike = mindspore.ops.ZerosLike()

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(oneslike(sr), sigma(sr - mindspore.ops.ReduceMean(hr))) +
            cross_entropy(zeroslike(sr), sigma(hr -  mindspore.ops.ReduceMean(sr))))
    def generator_loss(hr, sr):
        return cross_entropy(oneslike(sr), sigma(sr))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
