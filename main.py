import paddle
import numpy as np
from paddle.nn import functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.transforms import ToTensor
import matplotlib.pyplot as plt


transform_tuple = Compose([
    ToTensor(),
    Normalize()
])

train_dataset = paddle.vision.MNIST(mode='train', transform=transform_tuple)
test_datasets = paddle.vision.MNIST(mode='test', transform=transform_tuple)




