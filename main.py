import paddle
from paddle import nn
from paddle import optimizer
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.transforms import ToTensor

import paddle.distributed as dist

from model_zoo import Model

transform_tuple = Compose([
    ToTensor(),
    Normalize()
])

parallel_flag = False


if __name__ == '__main__':

    if parallel_flag:
        dist.init_parallel_env()

    leakRelu_crossEntropy_adam = Model(
        transform_tuple,
        nn.LeakyReLU,
        nn.CrossEntropyLoss,
        optimizer.Adam
    )

    leakRelu_crossEntropy_adam.train()
    leakRelu_crossEntropy_adam.validate()

    relu_crossEntropy_sgd = Model(
        transform_tuple,
        nn.ReLU,
        nn.CrossEntropyLoss,
        optimizer.SGD
    )
    relu_crossEntropy_sgd.train()
    relu_crossEntropy_sgd.validate()

    leakReLuCrossEntropySgd = Model(
        transform_tuple,
        nn.LeakyReLU,
        nn.CrossEntropyLoss,
        optimizer.SGD
    )

    leakReLuCrossEntropySgd.train()
    leakReLuCrossEntropySgd.validate()

    batch_size_256_leakRelu_crossEntropy_adam = leakRelu_crossEntropy_adam = Model(
        transform_tuple,
        paddle.nn.LeakyReLU,
        nn.CrossEntropyLoss,
        optimizer.Adam,
        batch_size=256
    )
    batch_size_256_leakRelu_crossEntropy_adam.train()
    batch_size_256_leakRelu_crossEntropy_adam.validate()

