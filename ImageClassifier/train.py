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
def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Defines the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loads the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, defines the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader

data_dir = '/home/workspace/ImageClassifier/flowers'
trainloader, validloader, testloader = transform_data(data_dir)

parser = argparse.ArgumentParser(description = 'Image Classifier training')

## Imports Model and defines hiddent layers
def create_model(architecture, hidden_layers, dropout = .02, pretrained=True, device_type=None):
    # Imports directories. 
    data_dir = 'flowers'
    train_dir = data_dir + '/train'

    # Selects Model to be imported for training
    model = getattr(models, architecture)(pretrained=pretrained)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
            
    # Creates an instance of the ImageFolder dataset
    image_datasets = torchvision.datasets.ImageFolder(train_dir)
    # Accesses the mapping of class indices to class labels
    class_to_idx = image_datasets.class_to_idx
    # Sets the mapping of class indices to class labels in the model
    model.class_to_idx = class_to_idx

    # Adjusts number of Hidden Layers
    if architecture == 'vgg16':
        input_size = model.classifier[0].in_features
    elif architecture == 'alexnet':
        input_size = model.classifier[1].in_features
    else:
        raise ValueError("Invalid architecture")
        
    output_size = len(class_to_idx)

    hidden_sizes = [input_size] + hidden_layers + [output_size]

    classifier = nn.Sequential()

    for i in range(len(hidden_sizes) - 1):
        classifier.add_module('fc{}'.format(i + 1), nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            classifier.add_module('relu{}'.format(i + 1), nn.ReLU())
             #Add L2 regularization
        classifier.add_module('dropout{}'.format(i + 1), nn.Dropout(p=dropout))  # adjust the dropout rate as needed


    classifier.add_module('output', nn.LogSoftmax(dim=1))

    model.classifier = classifier

    # Adjusts Model to run with either cuda or cpu as available
    if device_type is not None:
        device = torch.device(device_type)
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, class_to_idx

#architecture can be alexnet or vgg16
parser.add_argument('--arch', type=str, default = 'vgg16', help='Model architecture (vgg16 or alexnet)')
parser.add_argument('--hidden_layers', type=int, nargs='+', default=[624], help='Hidden layer sizes (default: [624])')
parser.add_argument('--dropout', type=float, default=0.02, help='Dropout rate (default: 0.02)')

#Begins Training Model
def train_model(model, trainloader, validloader, criterion, optimizer, epochs, print_every, gradient_accumulation_steps, device_type = None, verbose=True):
    if device_type is not None:
        device = torch.device(device_type)
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    steps = 0
    running_loss = 0
    test_loss = 0  
    accuracy = 0

    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            loss = loss / gradient_accumulation_steps  # Scale the loss

            loss.backward()

            if steps % gradient_accumulation_steps == 0:
                optimizer.step()

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
                        
            if verbose:
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
                
# Adjusts hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=0, help='Number of epochs (default: 0)')
parser.add_argument('--print_every', type=int, default=5, help='Print frequency (default: 5)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps (default: 4)')

#Saves Checkpoint
def save_checkpoint(model, class_to_idx, optimizer, filename, save_enabled=True):
    if not save_enabled:
        return
    
    checkpoint = {'model': model,
                  'class_to_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    if filename:
        torch.save(checkpoint, filename)

# Activates Checkpoint function if not commented out
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Checkpoint save directory (default: checkpoint.pth)')
parser.add_argument('--save_enabled', action='store_true', help='Enable checkpoint saving')
args = parser.parse_args()

architecture = args.arch
hidden_layers = args.hidden_layers
dropout = args.dropout
learning_rate = args.learning_rate
epochs = args.epochs
print_every = args.print_every
gradient_accumulation_steps = args.gradient_accumulation_steps

# Loads the model
model, class_to_idx = create_model(architecture, hidden_layers, dropout, pretrained=True)
# Create optimizer and criterion
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
# Call the train_model function with the adjusted hyperparameters
train_model(model, trainloader, validloader, criterion, optimizer, epochs, print_every, gradient_accumulation_steps, verbose=True)

save_checkpoint(model, class_to_idx, optimizer, 'checkpoint.pth')

#Example Command Line Code.  Remove '#'   
#python train.py --arch vgg16 --hidden_layers 512 256 --dropout 0.1 --learning_rate 0.001 --epochs 1 --print_every 1 --gradient_accumulation_steps 2 --save_enabled
