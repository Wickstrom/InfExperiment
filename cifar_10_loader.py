# %%
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

from VGG16 import VGG16
model = VGG16(10, nn.ReLU()).cuda()
#out, conv_layers = model(images)
#MI = []
#for i in range(len(conv_layers)):
#
#    MI.append(model.mutual_information(images, conv_layers[i]))
#
#print(MI)

# %%

import time


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs_gpu = inputs.cuda()
        labels_gpu = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, conv_layers = model(inputs_gpu)
        loss = criterion(outputs, labels_gpu)
        loss.backward()
        optimizer.step()
        if (i % 50) == 0:
            for j in range(len(conv_layers)):
                print(model.mutual_information(inputs, conv_layers[j].cpu()))
        
        time.sleep(5)

        if (i % 100) == 0:
            print(loss)

print('Finished Training')
