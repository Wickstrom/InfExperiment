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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
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


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    print('Epoch: ', epoch)
    running_loss = 0.0
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
        
#        if (i % 5) == 0:
#            print(i)
#        if (i % 10) == 0:
#            print("MI")
#            print(model.renyi(inputs))
#            for j in range(len(conv_layers)):
#                print(model.mutual_information(inputs, conv_layers[j].cpu()))
#                 print(model.renyi(conv_layers[j].cpu()))
        

                
                
        


        if (i % 100 and i != 0) == 0:
            print('Loss')
            print(loss)
#            dataiter = iter(testloader)
#            images, labels = dataiter.next()
#            outputs, test_conv= model(images.cuda())
#            _, predicted = torch.max(outputs, 1)
#            print((predicted == labels.cuda()).sum().item())

            mi_mat = np.zeros((17,17))
            
            for i, layer in enumerate(layers, 0):
                
                mi_mat[0, 0] = model.mutual_information(inputs, inputs)    
                mi_mat[0, i+1] = model.mutual_information(inputs, layer.cpu())
            
            for i in range(16):
                for j in range(i,16):
                    mi_mat[i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) 
            print(mi_mat)
print('Finished Training')
