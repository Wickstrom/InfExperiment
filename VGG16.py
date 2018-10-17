import scipy
import torch
import numpy as np
import torch.nn as nn
from time import sleep
import torch.nn.init as init
from numpy import linalg as LA
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

# Class for the VGG16 network.

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
        
        self.fc1 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(512, 4096),
                activation,]))
        self.fc2 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(4096, 4096),
                activation,]))
        self.classifier = nn.Sequential(*([
                nn.Linear(4096, num_classes),]))

        for m in self.modules(): # weight initialization.
            self.weight_init(m)

    def forward(self, x):                           # Forward pass for network.

        layers = []                                 # We collect the ouput of each layer in this list
        # Encoder 1
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool1(layer2)

        layers.append(layer1)                       # Append the ouput of the first and second convolutional layer.
        layers.append(layer2)

        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool2(layer4)

        layers.append(layer3)
        layers.append(layer4)

        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool3(layer7)

        layers.append(layer5)
        layers.append(layer6)
        layers.append(layer7)

        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool4(layer10)

        layers.append(layer8)
        layers.append(layer9)
        layers.append(layer10)

        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        layer13 = self.layer13(layer12)
        pool5 = self.pool4(layer13)

        layers.append(layer11)
        layers.append(layer12)
        layers.append(layer13)

        # Classifier

        fc1 = self.fc1(pool5.view(pool5.size(0), -1)) # Reshape from (N, C, H, W) to (N, C*H*W) form.
        fc2 = self.fc2(fc1)
        classifier = self.classifier(fc2)

        layers.append(fc1)
        layers.append(fc2)
        layers.append(classifier)

        return classifier, layers                     # Returns the ouput of the network (Softmax not applied yet!)
                                                      # and a list containing the output of each layer.

    def weight_init(self, m):                         # Function for initializing wieghts.
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
        if isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)

    def gram_matrix(self, x):                           # Calculate Gram matrix

#        sigma = 5*x.size(0)**(-1/(x.size(1)*x.size(2)))# Silverman's rule of Thumb

        if x.dim() == 2:                                # If the input is on matrix-form.
            k = x.data.numpy()
            k = squareform(pdist(k, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) # Calculate kernel width based on 10 nearest neighbours.
            k = scipy.exp(-k ** 2 / sigma ** 2)         # RBF-kernel.
            k = k / np.float64(np.trace(k))             # Normalize kernel.
        if x.dim() == 4:                                # If the input is on tensor-form
            k = x[:, 0].view(x.size(0), -1).data.numpy()
            k = squareform(pdist(k, 'euclidean'))
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            k = scipy.exp(-k ** 2 / sigma ** 2)
            for i in range(x.size(1)-1):                # Loop through all feature maps.
                k_temp = x[:, i+1].view(x.size(0), -1).data.numpy()
                k_temp = squareform(pdist(k_temp, 'euclidean'))
                sigma = np.mean(np.mean(np.sort(k_temp[:, :10], 1)))
                k = np.multiply(k, scipy.exp(-k_temp ** 2 / sigma ** 2)) # Multiply kernel matrices together.

            k = k / np.float64(np.trace(k)) # Normalize final kernel matrix.
        return k # Return the kernel martix of x

    def renyi(self, x): # Matrix formulation of Renyi entropy.
        alpha = 1.01
        k = self.gram_matrix(x) # Calculate Gram matrix.
        l, v = LA.eig(k)        # Eigenvalue/vectors.
        lambda_k = np.abs(l)    # Remove negative/im values.

        return (1/(1-alpha))*np.log2(np.sum(lambda_k**alpha)) # Return entropy.

    def joint_renyi(self, x, y): # Matrix formulation of joint Renyi entropy.
        alpha = 1.01
        k_x = self.gram_matrix(x)
        k_y = self.gram_matrix(y)
        k = np.multiply(k_x, k_y)
        k = k / np.float64(np.trace(k))

        l, v = LA.eig(k)
        lambda_k = np.abs(l)

        return (1/(1-alpha))*np.log2(np.sum(lambda_k**alpha))

    def mutual_information(self, x, y):  # Matrix formulation of mutual Renyi entropy.

        return self.renyi(x)+self.renyi(y)-self.joint_renyi(x, y)
