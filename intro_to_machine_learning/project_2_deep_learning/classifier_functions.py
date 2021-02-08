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

#######################################

def dataloaders(foldername):
    data_dir = foldername
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
#     print("Directories are defined")

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
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
#     data_transforms = {train_transforms, valid_transforms, test_transforms}
#     print("Transforms are defined")

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
#     image_datasets = {train_dataset, valid_dataset, test_dataset}
#     print("Datasets are defined")

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
#     dataloaders = {trainloader, validloader, testloader}
#     print("Dataloaders are defined")
    
    return train_transforms, valid_transforms, test_transforms, train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader

#######################################

def define_classifer(m, dropout, lr, hiddenunit):
    if m == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif m == 'vgg13':
        model= models.vgg13(pretrained= True)
        model.name= 'vgg13'
    else:
        print("Choose 'vgg16' or 'vgg13'")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088,hiddenunit)),
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(dropout)),
    ('fc4', nn.Linear(hiddenunit, 102)),
    ('log_softmax', nn.LogSoftmax(dim=1))]))   
#     model.classifier = nn.Sequential(nn.Linear(25088, 1024),
#                                      nn.ReLU(),
#                                      nn.Dropout(dropout),
#                                      nn.Linear(1024, 102),
#     #                                  nn.Linear(512, 102),
#                                      nn.LogSoftmax(dim=1))

    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(), lr= lr)
    

        
    return model, criterion, optimizer

#######################################

def train_model(model, train_dataset, trainloader, validloader, testloader, criterion, optimizer, epochs, resource,savedir):
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    if resource == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    print("=============== START TRAINING ===============")
    for epoch in range(epochs):        
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("=============== TRAINING FINISHED ===============")            
                
    test_accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        print("=============== START TESTING ===============")
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss = criterion(logps, labels)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    print("=============== TESTING FINISHED ===============")
        
    print("Saving checkpoint...")
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'architecture': 'vgg16',
                  'input_size': 25088, 
                  'output_size': 102,
                  'epochs': epochs,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'index': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'classindex': model.class_to_idx,
                  'learning_rate': 0.003}
    torch.save(checkpoint, savedir)
    print("Checkpoint save success")
        
    return

#######################################
