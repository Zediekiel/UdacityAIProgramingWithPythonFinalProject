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
    
    if architecture == 'vgg16':
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
    else: 
        None
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
        hidden_layers[-1] = input_size
    else:
        raise ValueError("Invalid architecture")

    output_size = len(class_to_idx)

    hidden_sizes = [input_size] + hidden_layers + [output_size]

    classifier = nn.Sequential()

    for i in range(len(hidden_sizes) - 1):
        classifier.add_module('fc{}'.format(i + 1), nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            classifier.add_module('relu{}'.format(i + 1), nn.ReLU())

    classifier.add_module('output', nn.LogSoftmax(dim=1))

    if architecture == 'vgg16':
        model.classifier = classifier
    elif architecture == 'alexnet':
        model.classifier[1] = classifier

    # Adjusts Model to run with either cuda or cpu as available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, class_to_idx