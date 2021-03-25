import mindspore
from mindspore import nn as nn
from src.archs.VGG import  vgg19
import mindspore.ops.functional as F
class TVLoss(nn.Cell):
    # 带正则化项的感知损失 Total Variation loss
    def __init__(self,weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def construct(self, X):
        X_size = mindspore.ops.Shape(X)
        batch_size = X_size[0]
        h_x = X_size[2]
        w_x = X_size[3]
        count_h = self.tensor_size(X[:, :, 1:, :])
        count_w = self.tensor_size(X[:, :, :, 1:])
        pow = mindspore.ops.Pow()
        sum = mindspore.ops.ReduceSum(keep_dims=True)
        h_tv = pow((X[:, :, 1:, :] - X[:, :, :h_x - 1, :]), 2)
        h_tv = sum(h_tv)
        w_tv = pow((X[:, :, :, 1:] - X[:, :, :, :w_x - 1]), 2)
        w_tv = sum(w_tv)
        tv_loss = self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return tv_loss

    @staticmethod
    def tensor_size(t):
        t_shape = mindspore.ops.Shape(t)
        return t_shape[1] * t_shape[2] * t_shape[3]


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

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
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
