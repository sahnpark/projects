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
import classifier_functions
import predict_functions
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Predict classification')
parser.add_argument('--data_dir', action="store", default="flowers", help="Enter directory folder name")
parser.add_argument('--img_dir', action="store", default="flowers/test/102/image_08042.jpg", help="Enter image")
parser.add_argument('--checkpoint_dir', action="store", default="checkpoint.pth", help="Enter checkpoint directory")
parser.add_argument('--topk', default=5, dest="topk", action="store", type=int)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", choices=["gpu", "cpu"])

parser.parse_args()
args = parser.parse_args()

    
def main():
    
    train_transforms, valid_transforms, test_transforms, train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader = classifier_functions.dataloaders(args.data_dir)
    
    model = predict_functions.load_checkpoint(args.checkpoint_dir)
    print(model)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    cat_sorted = json.dumps(cat_to_name, sort_keys=True)
    
    img = predict_functions.process_image(args.img_dir)
#     ax = predict_functions.imshow(img)
    prob, classes = predict_functions.predict(args.img_dir, model, args.topk, args.gpu)
    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes_name = list()
    for idx in np.arange(0,args.topk):
        classes_name.append(cat_to_name[classes[idx]])
#     print(prob)
#     print(classes)
    print('Top',args.topk, 'Classes:')
    for idx in np.arange(0,args.topk):
        print('#',idx+1,': ', classes_name[idx], '| Probability: ', prob[idx])

       

if __name__== "__main__":
    main()