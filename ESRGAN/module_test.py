from mindspore import context, Tensor
import numpy as np
import mindspore

Tensor1 = Tensor(np.random.random(size=(2,2,2)))
Tensor2 = Tensor1.reshape(-1)
print(Tensor2.shape)