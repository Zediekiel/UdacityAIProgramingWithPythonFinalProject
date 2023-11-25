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


#Imports and transforms data
from data_transformer import transform_data
data_dir = '/home/workspace/ImageClassifier/flowers'
trainloader, validloader, testloader = transform_data(data_dir)

## Imports Model and defines hiddent layers
from make_model import create_model
#archtiecture can be alexnet or vgg16
architecture = "alexnet"
hidden_layers = [400]

model, class_to_idx = create_model(architecture, hidden_layers, pretrained=True)


#Begins Training Model
from train_network import train_model
# Adjusts hyperparameters
learning_rate = 0.001
epochs = 1
print_every = 5
gradient_accumulation_steps = 4

# Create optimizer and criterion
#optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.NLLLoss()

# Call the train_model function with the adjusted hyperparameters
train_model(model, trainloader, validloader, criterion, optimizer, epochs, print_every, gradient_accumulation_steps)

#Saves Checkpoint
def save_checkpoint(model, class_to_idx, optimizer, filename):
    checkpoint = {'model': model,
                  'class_to_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    if filename:
        torch.save(checkpoint, filename)

# Activates Checkpoint function if not commented out
save_checkpoint(model, class_to_idx, optimizer, 'checkpoint.pth')
