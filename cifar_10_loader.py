# %%
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,         # Load Cifar10 training dataset.
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,        # Dataloader
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,        # Load Cifar10 test dataset.
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)     # Dataloader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
# I have commented out some lines which I just use during experimentation.

import numpy as np
#dataiter = iter(trainloader)
#images, labels = dataiter.next()


from VGG16 import VGG16             # Import network from VGG script
model = VGG16(10, nn.ReLU()).cuda() # Initialize model with 10 classes and relu activation function.
                                    # Note that if you run this scipt on a cpu you must remove the ".cuda()"
                                    # from this line.


#out, layers = model(images)

# Nevermind this part, it is just for prototyping and experimenting.
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

mi_mat = np.zeros((1, 17, 17))   # Mutual information matrix (just zeros for now).
j_mat = np.zeros((1, 17, 17))    # Joint information matrix (just zeros for now).
dataiter = iter(testloader)      # Data iterator, for running through the data.
inputs, labels = dataiter.next()
outputs, layers = model(inputs.cuda()) # Get the ouput of the network and each layer before any training.

for i, layer in enumerate(layers, 0): # Loop through the output of each layer.

    mi_mat[0, 0, 0] = model.mutual_information(inputs, inputs)       # MI between X and X
    mi_mat[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())# Mi between X and all other feature maps.

    j_mat[0, 0, 0] = model.joint_renyi(inputs, inputs)               # Joint between X and X
    j_mat[0, 0, i+1] = model.joint_renyi(inputs, layer.cpu())        # Joint between X and all other feature maps.

for i in range(16): # Loop through all layers. Maybe these two loops can be combined.
    for j in range(i, 16):
        mi_mat[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) # MI between all feature maps.
        j_mat[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) # Joint between all feature maps.

criterion = nn.CrossEntropyLoss()               # Cross-entropy loss
optimizer = torch.optim.Adam(model.parameters())# ADAM optimizer.

cost = [] # List for keeping track of training error
test = [] # list for keeping track of test error.

for epoch in range(10):     # Run for 10 epochs
    print('Epoch: ', epoch) 
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    outputs, layers = model(inputs.cuda())

    mi_mat_temp = np.zeros((1, 17, 17)) # Temporary matrices for stroing MI and joint.
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
            
    if epoch != 0: # We are not adding the temp matrices to the actual matrix for the first epoch, 
        mi_mat = np.concatenate((mi_mat, mi_mat_temp)) # since we already calcualted this outside the loop.
        j_mat = np.concatenate((j_mat, j_mat_temp))

    for i, data in enumerate(trainloader, 0): # Run through entire training set.

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, layers = model(inputs.cuda())  # If you wnat to run on cpu
        loss = criterion(outputs, labels.cuda())# remove ".cuda()"
        loss.backward()
        optimizer.step()
        cost_temp.append(loss.cpu().data.numpy()) # Append training cost
    cost.append(np.mean(cost_temp))               # Take mean cost of these batches
    print(cost[-1])              


print('Finished Training')
np.savez_compressed('MI_cifar_10', a=mi_mat, b=j_mat, c=cost) # Save the final MI matrix, Joint matrix and cost as
                                                              # "MI_cifar_10.npz"
