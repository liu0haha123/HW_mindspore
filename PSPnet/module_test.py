from src.model.PSPnet import  ResNet
import mindspore
import numpy as np
from mindspore import context, Tensor

model = ResNet(pretrained_path=None)

test_Tensor = Tensor(np.random.rand(1,3, 473,473), dtype=mindspore.float32)
out = model(test_Tensor)
print(out)