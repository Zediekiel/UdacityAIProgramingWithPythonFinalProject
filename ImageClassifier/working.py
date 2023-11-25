import torch
from torchvision import models
import torch.nn as nn
import torchvision

def create_model(architecture, hidden_layers, pretrained=True):
    # Imports directories. Additional
    data_dir = 'flowers'
    train_dir = data_dir + '/train'

    # Selects Model to be imported for training
    model = getattr(models, architecture)(pretrained=pretrained)
    # Creates an instance of the ImageFolder dataset
    image_datasets = torchvision.datasets.ImageFolder(train_dir)
    # Accesses the mapping of class indices to class labels
    class_to_idx = image_datasets.class_to_idx
    # Sets the mapping of class indices to class labels in the model
    model.class_to_idx = class_to_idx

    for param in model.parameters():
        param.requires_grad = False

    # Adjusts number of Hidden Layers
    input_size = model.classifier[0].in_features
    output_size = len(class_to_idx)

    hidden_sizes = [input_size] + hidden_layers + [output_size]

    classifier = nn.Sequential()

    for i in range(len(hidden_sizes) - 1):
        classifier.add_module('fc{}'.format(i + 1), nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            classifier.add_module('relu{}'.format(i + 1), nn.ReLU())

    classifier.add_module('output', nn.LogSoftmax(dim=1))

    model.classifier = classifier

    # Adjusts Model to run with either cuda or cpu as available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


######

#Retrieve when train.py is fixed
#Begins Training Model
from train_network import train_model
# Adjusts hyperparameters
learning_rate = 0.0001
epochs = 1
print_every = 5
gradient_accumulation_steps = 4

# Create optimizer and criterion
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
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


##
import torch
from torchvision import models
import torch.nn as nn
import torchvision

def create_model(architecture, hidden_layers, pretrained=True):
    # Imports directories. Additional 
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    
    # Selects Model to be imported for training
    model = getattr(models, architecture)(pretrained=pretrained)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Creates an instance of the ImageFolder dataset
    image_datasets = torchvision.datasets.ImageFolder(train_dir)
    # Accesses the mapping of class indices to class labels
    class_to_idx = image_datasets.class_to_idx
    # Sets the mapping of class indices to class labels in the model
    model.class_to_idx = class_to_idx
    
    # Checks if input model has a classifier or fc attribute
    if hasattr(model, 'classifier'):
        input_size = model.classifier[0].in_features
    elif hasattr(model, 'fc'):
        input_size = model.fc.in_features
    else:
        raise AttributeError("The model does not have a classifier or fc attribute.")
    
    # Adjusts number of Hidden Layers    
    output_size = len(class_to_idx)
    
    hidden_sizes = [input_size] + hidden_layers + [output_size]
    
    classifier = nn.Sequential()
    
    for i in range(len(hidden_sizes)-1):
        classifier.add_module('fc{}'.format(i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        if i < len(hidden_sizes)-2:
            classifier.add_module('relu{}'.format(i+1), nn.ReLU())
    
    classifier.add_module('output', nn.LogSoftmax(dim=1))
    
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    elif hasattr(model, 'fc'):
        model.fc = classifier
    
    # Adjusts Model to run with either cuda or cpu as available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, class_to_idx