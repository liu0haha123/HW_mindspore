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
from src.utils.metrics import SoftmaxCrossEntropyLoss

set_seed(1)
np.random.seed(1)


class Aux_CELoss_Cell(nn.Cell):
    def __init__(self, network, num_classes, ignore_label):
        super(Aux_CELoss_Cell, self).__init__()
        self.loss = SoftmaxCrossEntropyLoss(num_classes, ignore_label)
        self.network = network

    def construct(self, image, target):
        predict_aux, predict = self.network(image)
        CE_loss = self.loss(predict, target)
        CE_loss_aux = self.loss(predict_aux, target)
        loss = CE_loss + (0.4 * CE_loss_aux)

        return loss


class WithLossCell(nn.Cell):
    def __init__(self, network):
        super(GeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, image, target):
        loss = self.network(image, target)
        return loss


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, image, target):
        weights = self.weights
        loss = self.network(image, target)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(image, target, sens)
        return ops.depend(loss, self.optimizer(grads))
