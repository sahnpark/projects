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
import classifier_functions
import json

parser = argparse.ArgumentParser(description='Train data')


parser.add_argument('--data_dir', action="store", default="flowers", help="Enter directory folder name")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", choices=["vgg16", "vgg13"])
parser.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.003)
parser.add_argument('--dropout', dest = "dropout", action = "store", type=float, default = 0.0)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", type=int, default=512)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", choices=["gpu", "cpu"])

parser.parse_args()
args = parser.parse_args()


    
def main():
    
    train_transforms, valid_transforms, test_transforms, train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader = classifier_functions.dataloaders(args.data_dir)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model, criterion, optimizer = classifier_functions.define_classifer(args.arch, args.dropout, args.learning_rate, args.hidden_units)
    classifier_functions.train_model(model, train_dataset, trainloader, validloader, testloader, criterion, optimizer, args.epochs, args.gpu, args.save_dir)

if __name__== "__main__":
    main()