# Imports
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
#Import json file for category labels for images
import json


#Load Checkpoint
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

def load_checkpoint(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width, height = image.size
    
    aspect_ratio = width/height
    
    if width < height:
        target_size = (256, int(256 / aspect_ratio))
    else: 
        target_size = (int(256 * aspect_ratio), 256)
    
    pil_image = image.resize(target_size)
    center_pil_image = transforms.CenterCrop(224)(pil_image)
    np_image = np.array(center_pil_image)

    # Normalize the image
    np_image = np_image / 255.0
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])
    
    # Transpose the image array
    np_image = np_image.transpose((2, 0, 1))
    return np_image

#Isolates image to be selected for prediction
image_dir = 'flowers/test/1/image_06743.jpg'
img = Image.open(image_dir)

#Predict top K Classes
# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Load the mapping of class indices to class names from cat_to_name.json  
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    processed_image = process_image(image_path)
    
    image_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    model.eval()
    
# Disable gradient calculation
    with torch.no_grad():
        # Forward pass through the model
        output = model(image_tensor)
        
        # Calculate the probabilities by applying softmax
        probabilities = torch.exp(output)
        
        # Get the top k probabilities and classes
        top_probabilities, top_classes = probabilities.topk(topk)
        
        # Convert the tensors to lists
        top_probabilities = top_probabilities.squeeze().tolist()
        top_classes = top_classes.squeeze().tolist()
        
        # Convert the indices to class labels
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_labels = [cat_to_name[idx_to_class[idx]] for idx in top_classes]
        # Create a paired list of top labels and top probabilities
        output = list(zip(top_labels, top_probabilities))
        
        return output
result = predict(img, load_checkpoint('checkpoint.pth'))
print(result)