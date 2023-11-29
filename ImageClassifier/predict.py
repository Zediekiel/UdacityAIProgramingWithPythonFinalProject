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
import os
#Imports argument parser
import argparse

#Load Checkpoint
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

def load_checkpoint(filepath, device_type = None):
    if device_type is not None:
        device = torch.device(device_type)
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    #model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    
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

#Predict top K Classes
def predict(image_path, model, topk=5, category_names='cat_to_name.json', device_type=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the cat_to_name.json file
    category_names_path = os.path.join(script_dir, args.category_names)
    
    # Load the mapping of class indices to class names from the input .json file  
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    processed_image = process_image(image_path)
    
    image_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
    
    if device_type is not None:
        device = torch.device(device_type)
    else: 
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
        
        # Reverse the class-to-idx dictionary to create idx_to_class dictionary
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        
        # Convert the predicted indices to class names
        predicted_classes = [idx_to_class[idx] for idx in top_classes]
        
        # Convert the indices to class labels
        if cat_to_name is not None:
            top_labels = [cat_to_name[idx_to_class[idx]] for idx in top_classes]
        else:
            top_labels = predicted_classes
        
        # Create a paired list of top labels and top probabilities
        output = list(zip(top_labels, top_probabilities))
        
        return output
   
#Define aruments
parser = argparse.ArgumentParser(description='Image Classifier Prediction')
parser.add_argument('image_path', type=str, help='Path to the image')
parser.add_argument('model_path', type=str, help='Path to the model checkpoint')
parser.add_argument('--topk', type=int, default=5, help='Top K most likely classes (default: 5)')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names mapping file (default: cat_to_name.json)')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for prediction (default: cuda)')
args = parser.parse_args()
   
#Loads model    
model = load_checkpoint(args.model_path, device_type=args.device)
    
# Call the predict function
predictions = predict(args.image_path, model, args.topk, args.category_names, device_type=args.device)

# Print the top K classes
for label, probability in predictions:
    print(f"Class: {label}, Probability: {probability}")
    
#Example Command Line Code.  Remove '#'   
#python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoint.pth' --topk 5 --category_names 'cat_to_name.json' --device cpu

 
