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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=300,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%

import numpy as np
#dataiter = iter(trainloader)
#images, labels = dataiter.next()


from VGG16 import VGG16
model = VGG16(10, nn.ReLU()).cuda()
#out, layers = model(images)


# %%

#x = conv_layers[0]
#k = model.gram_matrix(x[:, 0, :, :])
#
#for i in range(x.size(1)-1):
#    k = np.multiply(k, model.gram_matrix(x[:, i+1, :, :]))
#    k = k /np.trace(k)
#l, v = LA.eig(k)
#lambda_x = np.abs(l)

# %%

mi_mat = np.zeros((1, 17, 17))
dataiter = iter(testloader)
inputs, labels = dataiter.next()
outputs, layers = model(inputs.cuda())

for i, layer in enumerate(layers, 0):

    mi_mat[0, 0, 0] = model.mutual_information(inputs, inputs)
    mi_mat[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())

for i in range(16):
    for j in range(i,16):
        mi_mat[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2):
    print('Epoch: ', epoch)
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    outputs, layers = model(inputs.cuda())

    mi_mat_temp = np.zeros((1, 17, 17))
    
    for i, layer in enumerate(layers, 0):
        
        mi_mat_temp[0, 0, 0] = model.mutual_information(inputs, inputs)    
        mi_mat_temp[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())
    
    for i in range(16):
        for j in range(i,16):
            mi_mat_temp[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) 
            
    if epoch != 0:
        mi_mat = np.concatenate((mi_mat, mi_mat_temp))

    for i, data in enumerate(trainloader, 0):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, layers = model(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()


print('Finished Training')
np.savez_compressed('MI_cifar_10', a=mi_mat)
