import numpy as np
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.parallel._utils import (
    _get_device_num,
    _get_gradients_mean,
    _get_parallel_mode,
)
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.common import initializer as init, set_seed
from src.model.loss import PerceptualLoss
set_seed(1)
np.random.seed(1)

class GeneratorWrapLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    """
    def __init__(self, network):
        super(GeneratorWrapLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, lr, hr, fake_labels, real_labels):
        _, G_Loss, _, _, _, = self.network(lr, hr, fake_labels, real_labels)
        return G_Loss


class DiscriminatorWrapLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    """
    def __init__(self, network):
        super(DiscriminatorWrapLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, hr_fake, hr):
        D_Loss = self.network(hr_fake, hr)
        return D_Loss

class GeneratorLossCell(nn.Cell):
    def __init__(self, generator, discriminator,pretrained_path):
        super(GeneratorLossCell,self).__init__()
        self.perception_criterion = PerceptualLoss(pretrained_path)
        self.content_criterion = nn.L1Loss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.G = generator
        self.D = discriminator
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, lr, hr, fake_labels, real_labels):
        fake_hr = self.G(lr)
        score_real = self.D(hr)
        score_fake = self.D(fake_hr)
        discriminator_rf = score_real - self.mean(score_fake)
        discriminator_fr = score_fake - self.mean(score_real)
        adversarial_loss_rf = self.adversarial_criterion(
            discriminator_rf, fake_labels)
        adversarial_loss_fr = self.adversarial_criterion(
            discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
        perceptual_loss = self.perception_criterion(hr, fake_hr)
        content_loss = self.content_criterion(hr, fake_hr)
        generator_loss = (
            5e-3 * adversarial_loss
            + 1.0 * perceptual_loss
            + 1e-1 * content_loss
        )
        return (fake_hr, generator_loss, content_loss, perceptual_loss, adversarial_loss)


class DiscriminatorLossCell(nn.Cell):
    def __init__(self, discriminator):
        super(DiscriminatorLossCell, self).__init__()
        self.D = discriminator
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, hr_fake, hr):
        score_real = self.D(hr)
        score_fake = self.D(hr_fake)
        discriminator_rf = score_real - self.mean(score_fake)
        discriminator_fr = score_fake - self.mean(score_real)
        discriminator_loss = (discriminator_fr + discriminator_rf) / 2

        return discriminator_loss


class TrainOneStepCellGen(nn.Cell):
    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOneStepCellGen, self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = GeneratorWrapLossCell(G)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        ):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                self.weights, mean, degree)

    def construct(self, lr, hr, fake_labels, real_labels):
        weights = self.weights
        hr_fake,generator_loss,content_loss,perception_loss,adversarial_loss = self.G(lr, hr, fake_labels, real_labels)
        sens = P.Fill()(P.DType()(generator_loss), P.Shape()(generator_loss), self.sens)
        grads = self.grad(self.network, weights)(lr, hr, fake_labels, real_labels, sens)
        grads = self.grad_reducer(grads)

        return hr_fake,F.depend(generator_loss, self.optimizer(grads))
        

class TrainOneStepCellDis(nn.Cell):
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepCellDis, self).__init__()
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DiscriminatorWrapLossCell(D)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        ):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                self.weights, mean, degree)

    def construct(self, hr_fake, hr):
        weights = self.weights
        loss = self.D(hr_fake, hr)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(hr_fake, hr, sens)
        grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads))

