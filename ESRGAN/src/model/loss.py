import mindspore
from mindspore import nn as nn
from src.model.VGG import vgg16
import mindspore.ops.functional as F

class PerceptualLoss(nn.Cell):
    # 内容损失
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16()
        loss_network = nn.SequentialCell(*list(vgg.layers)[0:44])
        for l in loss_network.layers:
            l.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def construct(self, high_resolution, fake_high_resolution):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 读取时放缩到了0-1 在计算内容损失时应该还原
        # 12.75 is rescale factor for vgg featuremaps.
        perception_loss = self.l1_loss((self.loss_network(high_resolution* 255.)/12.75), (self.loss_network(fake_high_resolution* 255.)/12.75))
        return perception_loss

class DiscriminatorLoss(nn.Cell):
    def __init__(self,gan_type='ragan'):
        super(DiscriminatorLoss, self).__init__()
        self.gan_type = gan_type
        self.cross_entropy = mindspore.ops.BinaryCrossEntropy
        self.sigma = mindspore.ops.Sigmoid

    def construct(self, hr,sr):
        # hr是真实的高分辨率图像 sr是生成图像
        if self.gan_type =="ragan":
            return 0.5 * (
                    self.cross_entropy(F.ones_like(hr), self.sigma(hr - mindspore.ops.ReduceMean(sr))) +
                    self.cross_entropy(F.zeros_like(sr), self.sigma(sr - mindspore.ops.ReduceMean(hr))))
        elif self.gan_type == 'gan':
            real_loss = self.cross_entropy(F.ones_like(hr), self.sigma(hr))
            fake_loss = self.cross_entropy(F.zeros_like(sr), self.sigma(sr))
            return real_loss + fake_loss

class GeneratorLoss(nn.Cell):
    def __init__(self,gan_type='ragan'):
        super(GeneratorLoss, self).__init__()
        self.gan_type = gan_type
        self.cross_entropy = mindspore.ops.BinaryCrossEntropy
        self.sigma = mindspore.ops.Sigmoid

    def construct(self, hr,sr):
        # hr是真实的高分辨率图像 sr是生成图像
        if self.gan_type =="ragan":
            return 0.5 * (
                    self.cross_entropy(F.ones_like(sr), self.sigma(sr - mindspore.ops.ReduceMean(hr))) +
                    self.cross_entropy(F.zeros_like(hr), self.sigma(hr - mindspore.ops.ReduceMean(sr))))
        elif self.gan_type == 'gan':
            return self.cross_entropy(F.ones_like(sr),self.sigma(sr))
