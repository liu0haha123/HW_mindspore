from mindspore import context, Tensor
import numpy as np
import mindspore
from src.model.loss import PerceptualLoss, DLoss


def test():
    test_input = Tensor(np.random.random(size=(20, 100)), dtype=mindspore.float32)
    test_out = Tensor(np.random.random(size=(20, 100)), dtype=mindspore.float32)
    loss = DLoss()
    out = loss(test_input, test_out)
    print(out)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2)
    test()
