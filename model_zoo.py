# coding:utf-8
from paddle import nn
from paddle.io import DataLoader

import paddle

from visualdl import LogWriter
import os

import matplotlib.pyplot as plt
import numpy as np


class Model():

    def __init__(self, transforms, activate_func, criterion, optimizer, epochs=20, batch_size=128, is_parallel=False):
        self.activate_function = activate_func()
        self.criterion = criterion()

        self.epochs = epochs

        self.losses = []
        self.acces = []
        self.eval_losses = []
        self.eval_acces = []

        # net = nn.Sequential(
        #             paddle.nn.Flatten(),
        #             paddle.nn.Linear(784 * 3, 512),
        #             self.activate_function,
        #             paddle.nn.Dropout(0.2),
        #             paddle.nn.Linear(512, 10)
        #         )
        net = nn.Sequential(
            paddle.nn.Flatten(),
            nn.Linear(784 * 3, 400),
            self.activate_function,
            nn.Linear(400, 200),
            self.activate_function,
            nn.Linear(200, 100),
            self.activate_function,
            nn.Linear(100, 10)
        )
        if is_parallel:
            self.net = paddle.DataParallel(net)
        else:
            self.net = net

        self.optm = optimizer(parameters=self.net.parameters(), learning_rate=1e-1)

        train_dataset = paddle.vision.FashionMNIST(mode='train', transform=transforms)
        test_datasets = paddle.vision.FashionMNIST(mode='test', transform=transforms)


        # train_dataset = paddle.vision.MNIST(mode='train', transform=transforms)
        # test_datasets = paddle.vision.MNIST(mode='test', transform=transforms)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

        self.str_config = type(self.activate_function).__name__ + '-' + type(self.criterion).__name__ + '-' + type(self.optm).__name__

    def train(self):
        idx = 0

        print(self.str_config)

        with LogWriter(logdir= os.path.join('./log', self.str_config, 'train')) as writer:
            for epoch in range(self.epochs):
                train_loss = 0
                train_acc = 0
                self.net.train()
                for batch_id, data in enumerate(self.train_loader()):
                    x_data = data[0]
                    # print('Type: ', type(data[1]))

                    # y_data = paddle.to_tensor(np.where(data[1].numpy() > 5, 1, 0))
                    y_data = paddle.to_tensor(data[1].numpy().astype(np.int64))
                    predicts = self.net(x_data)

                    train_loss = self.criterion(predicts, y_data)

                    train_acc = paddle.metric.accuracy(predicts, y_data)

                    train_loss.backward()

                    if (batch_id + 1) % 100 == 0:

                        print(
                            "epoch: {}, batch_id: {}, loss is: {}, acc is: {}"
                                .format(epoch + 1, batch_id + 1, train_loss.numpy(), train_acc.numpy()))

                        writer.add_scalar(tag='loss', step=idx, value=train_loss.numpy())
                        writer.add_scalar(tag='acc', step=idx, value=train_acc.numpy())

                        idx += 1

                    # self.losses.append(train_loss.numpy())
                    # self.acces.append(train_acc.numpy())

                    self.optm.step()
                    self.optm.clear_grad()

    def validate(self):
        # testingzX
        print('Validate: ')
        eval_loss = 0
        eval_acc = 0
        idx = 0
        self.net.eval()
        with LogWriter(logdir= os.path.join('./log', self.str_config, 'val')) as witer:
            for batch_id, data in enumerate(self.test_loader()):
                x = data[0]
                # y = paddle.to_tensor(np.where(data[1].numpy() > 5, 1, 0))
                y = data[1]
                predicts = self.net(x)
                eval_loss = self.criterion(predicts, y)
                eval_acc = paddle.metric.accuracy(predicts, y)

                # 打印信息
                if (batch_id + 1) % 100 == 0:
                    print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id + 1, eval_loss.numpy(), eval_acc.numpy()))

                    witer.add_scalar(tag='acc', step=idx, value=eval_acc.numpy())
                    witer.add_scalar(tag='loss', step=idx, value=eval_loss.numpy())

                    idx += 1

                # self.eval_acces.append(eval_acc.numpy())
                # self.eval_losses.append(eval_loss.numpy())
