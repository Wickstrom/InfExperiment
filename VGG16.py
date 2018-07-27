import scipy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from numpy import linalg as LA
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


class VGG16(nn.Module):
    def __init__(self, num_classes, activation):
        super(VGG16, self).__init__()

        # First encoder
        self.layer1 = nn.Sequential(
                *([nn.Conv2d(3, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   activation, ]))
        self.layer2 = nn.Sequential(
                *([nn.Conv2d(64, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   activation, ]))
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second encoder
        self.layer3 = nn.Sequential(
                *([nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   activation, ]))
        self.layer4 = nn.Sequential(
                *([nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   activation, ]))
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third encoder
        self.layer5 = nn.Sequential(
                *([nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation, ]))
        self.layer6 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation, ]))
        self.layer7 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation, ]))
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth encoder
        self.layer8 = nn.Sequential(
                *([nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.layer9 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.layer10 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fifth encoder
        self.layer11 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.layer12 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.layer13 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation, ]))
        self.pool5 = nn.MaxPool2d(2, 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            activation,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            activation,
            nn.Linear(4096, num_classes),
            )
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):                           # Forward pass for network.

        conv_layers = []
        # Encoder 1
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool1(layer2)

        conv_layers.append(layer1)
        conv_layers.append(layer2)

        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool2(layer4)

        conv_layers.append(layer3)
        conv_layers.append(layer4)

        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool3(layer7)

        conv_layers.append(layer5)
        conv_layers.append(layer6)
        conv_layers.append(layer7)

        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool4(layer10)

        conv_layers.append(layer8)
        conv_layers.append(layer9)
        conv_layers.append(layer10)

        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        layer13 = self.layer13(layer12)
        pool5 = self.pool4(layer13)

        conv_layers.append(layer11)
        conv_layers.append(layer12)
        conv_layers.append(layer13)

        # Classifier
        out = pool5.view(pool5.size(0), -1)

        return self.classifier(out), conv_layers

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
        if isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)

    def gram_matrix(self, x):

#        sigma_x = 5*x.size(0)**(-1/(x.size(1)*x.size(2)))
        x = x.view(x.size(0), -1).data.numpy()
        x = squareform(pdist(x, 'euclidean'))
#        sigma_x = np.mean(np.mean(np.sort(x[:, :10], 1)))
#        x = (1/x.shape[0])*scipy.exp(-x ** 2 / sigma_x ** 2)
        return x

    def renyi(self, x):
        alpha = 1.01
        gram_m = self.gram_matrix(x)
        l, v = LA.eig(gram_m)
        lambda_x = np.abs(l)

        return (1/(1-alpha))*np.log2(np.sum(lambda_x**alpha))

    def joint_renyi_conv(self, x):
        alpha = 1.01
        #length = np.random.choice(x.size(1), 15, replace=False)
        k = self.gram_matrix(x[:, 0, :, :])
        for i in range(3):
            k = np.multiply(k, self.gram_matrix(x[:, i+1, :, :]))
            k = k / np.trace(k)
        l, v = LA.eig(k)
        lambda_x = np.abs(l)
        print(lambda_x)

        return (1/(1-alpha))*np.log2(np.sum(lambda_x**alpha))

    def joint_renyi_all(self, x, y):
        alpha = 1.01
        #length = np.random.choice(y.size(1), 15, replace=False)
        k = self.gram_matrix(x)
        for i in range(3):
            k = np.multiply(k, self.gram_matrix(y[:, i, :, :]))
            k = k / np.trace(k)
        l, v = LA.eig(k)
        lambda_x = np.abs(l)

        return (1/(1-alpha))*np.log2(np.sum(lambda_x**alpha))

    def mutual_information(self, x, y):
        return (self.renyi(x) +
                self.joint_renyi_conv(y) -
                self.joint_renyi_all(x, y))
