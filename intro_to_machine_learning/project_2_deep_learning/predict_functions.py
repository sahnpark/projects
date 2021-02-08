import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import time
from PIL import Image
import argparse
import matplotlib.pyplot as plt

#######################################

def load_checkpoint(path):
    checkpoint= torch.load(path)
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['classindex']
#     optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    return model #, optimizer

#######################################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.thumbnail((256, 256))
    width, height = img.size
    
    #crop the center 224x224 portion
    
    left = (width-224)/2
    right = left+224
    top = (height-224)/2
    bottom = top+224
    img = img.crop((left, top, right, bottom))
    
    #normalize 
    np_img = np.array(img)/255
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - img_mean) / img_std
    
    np_img = np_img.transpose(2,0,1)
    img = torch.from_numpy(np_img).float()
    return img

#######################################

# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None#         fig, ax = plt.subplots()
    
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image)
    
#     return ax

#######################################


def predict(image_path, model, topk, resource):


    if torch.cuda.is_available() and resource =='gpu':
        model.to('cuda')
    img=process_image(image_path)
    img= img.unsqueeze_(0)
    img= img.float()
        
    if resource == 'gpu':
        with torch.no_grad():
            logps= model.forward(img.to('cuda'))
    else:
        with torch.no_grad():
            logps= model.forward(img)
    
    p= torch.exp(logps.data)
    top_p, top_class = p.topk(topk)
    top_p = np.array(top_p.detach())[0] 
    top_class = np.array(top_class.detach())[0]
    
    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = list()
    for label in top_class:
        classes.append(class_idx_dict[label])
        
    return top_p, classes
