#For Plotting
import matplotlib.pyplot as plt
#For building NN
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import time
#For Image Display and transformation
from PIL import Image
import numpy as np
import json
#Saving NN
import argparse


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Defines the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

#Loads the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

#Using the image datasets and the trainforms, defines the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)                                      
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

#Select Model to be imported for training
model = models.vgg11(pretrained=True)
model

# Create an instance of the ImageFolder dataset
image_datasets = torchvision.datasets.ImageFolder(train_dir)
# Access the mapping of class indices to class labels
class_to_idx = image_datasets.class_to_idx
# Set the mapping of class indices to class labels in the model
model.class_to_idx = class_to_idx

for param in model.parameters():
    param.requires_grad = False
    
#Adjust number of Hidden Layers    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 400)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(400, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# Comment or uncomment code for use of cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
#device = torch.device("cpu")
model.to(device)

# Network for Training 

#Adjust Learning rate, number of epochs and hyperparameters
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = nn.NLLLoss()
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    gradient_accumulation_steps = 10
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
    
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
    
        loss = loss / gradient_accumulation_steps  # Scale the loss
    
        loss.backward()
    
        if steps % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
 
checkpoint = {'model': model, 'class_to_idx':class_to_idx, 'state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')
                            
