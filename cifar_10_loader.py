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
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
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
#
#mi_mat = np.zeros((1, 17, 17))
#j_mat = np.zeros((1, 17, 17))
#
#for i, layer in enumerate(layers, 0):
#
#    mi_mat[0, 0, 0] = model.mutual_information(images, images)
#    mi_mat[0, 0, i+1] = model.mutual_information(images, layer.cpu())
#    
#    j_mat[0, 0, 0] = model.joint_renyi(images, images)
#    j_mat[0, 0, i+1] = model.joint_renyi(images, layer.cpu())
#
#for i in range(16):
#    for j in range(i, 16):
#        mi_mat[0, i+1, j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu())
#        j_mat[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) 

# %%

mi_mat = np.zeros((1, 17, 17))
j_mat = np.zeros((1, 17, 17))
dataiter = iter(testloader)
inputs, labels = dataiter.next()
outputs, layers = model(inputs.cuda())

for i, layer in enumerate(layers, 0):

    mi_mat[0, 0, 0] = model.mutual_information(inputs, inputs)
    mi_mat[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())

    j_mat[0, 0, 0] = model.joint_renyi(inputs, inputs)
    j_mat[0, 0, i+1] = model.joint_renyi(inputs, layer.cpu())

for i in range(16):
    for j in range(i, 16):
        mi_mat[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu())
        j_mat[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

cost = []
test = []

for epoch in range(100):
    print('Epoch: ', epoch)
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    outputs, layers = model(inputs.cuda())

    mi_mat_temp = np.zeros((1, 17, 17))
    j_mat_temp = np.zeros((1, 17, 17))
    cost_temp = []
    
    for i, layer in enumerate(layers, 0):
        
        mi_mat_temp[0, 0, 0] = model.mutual_information(inputs, inputs)    
        mi_mat_temp[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())

        j_mat_temp[0, 0, 0] = model.joint_renyi(inputs, inputs)
        j_mat_temp[0, 0, i+1] = model.joint_renyi(inputs, layer.cpu())
    
    for i in range(16):
        for j in range(i,16):
            mi_mat_temp[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) 
            j_mat_temp[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) 
            
    if epoch != 0:
        mi_mat = np.concatenate((mi_mat, mi_mat_temp))
        j_mat = np.concatenate((j_mat, j_mat_temp))

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
        cost_temp.append(loss.cpu().data.numpy())
    cost.append(np.mean(cost_temp))
    print(cost[-1])


print('Finished Training')
np.savez_compressed('MI_cifar_10', a=mi_mat, b=j_mat, c=cost)
