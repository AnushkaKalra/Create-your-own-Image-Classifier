import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data

import os
import random
import json
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_transform(args):
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    test_dir = os.path.join(args.data_dir, "test")
        
    data_training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                  ])
    
    data_validation_transforms = transforms.Compose([transforms.Resize(255),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])
                                                    ])
    
    train_data = datasets.ImageFolder(train_dir, transform = data_training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = data_validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = data_validation_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return trainloader, validloader, testloader, train_data.class_to_idx

def model_training(args,trainloader, validloader, testloader, class_to_idx):
    
    if args.arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        args.hidden_units = 2048
    elif args.arch == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        args.hidden_units = 9216

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for param in model.parameters():
        param.requires_grad = False
        
    pretrained_in_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(in_features=pretrained_in_features, out_features=args.hidden_units, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p = 0.2),
                               nn.Linear(in_features=args.hidden_units, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    from torch import optim
    
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    
    # training
    # init variables for tracking loss/steps etc.
    epochs = args.epochs
    print_every = 5
    running_loss = running_accuracy = 0

    # for each epoch
    for e in range(0,epochs):
        step = 0
        running_loss = 0
        running_valid_loss = 0

        for images, labels in trainloader:
            step += 1

            model.train()

            # move images and model to device
            images, labels = images.to(device), labels.to(device)

            # zeroise grad
            optimizer.zero_grad()

            outputs = model.forward(images)    # forward

            trainloss = criterion(outputs, labels)
            trainloss.backward()               # backward

            optimizer.step()   # step

            running_loss += trainloss.item()

            # Calculating metrics
            ps = torch.exp(outputs)
            top_ps, top_class = ps.topk(1,dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # Resetting optimizer gradient & tracking metrics
            optimizer.zero_grad()
            running_loss += trainloss.item()
            running_accuracy += accuracy.item()

            if step % print_every == 0 or step == 1 or step == len(trainloader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e+1, epochs, (step)*100/len(trainloader)))

        # validate
        # turn model to eval mode
        # turn on no_grad

        model.eval()

        with torch.no_grad():
            # for each batch of images
            running_accuracy = 0
            running_loss = 0

            for images, labels in validloader:
            # move images and model to device
                images, labels = images.to(device), labels.to(device)

                outputs = model.forward(images)    # forward

                # loss
                valid_loss = criterion(outputs, labels)
                running_loss += valid_loss.item()

                # accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # print stats
        average_train_loss = running_loss/len(trainloader)
        average_valid_loss = running_valid_loss/len(validloader)
        accuracy = running_accuracy/len(validloader)
        print("Train Loss: {:.3f}".format(average_train_loss))
        print("Valid Loss: {:.3f}".format(average_valid_loss))
        print("Accuracy: {:.3f}%".format(accuracy*100))
        
        #Test set validation
        model.eval()
    
        with torch.no_grad():
            # for each batch of images
            running_accuracy = 0
            running_loss = 0

            for images, labels in testloader:
            # move images and model to device
                images, labels = images.to(device), labels.to(device)

                outputs = model.forward(images)    # forward

                # loss
                test_loss = criterion(outputs, labels)
                running_loss += test_loss.item()

                # accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        #average_test_loss = running_valid_loss/len(testloader)
        accuracy = running_accuracy/len(testloader)
        #print("Test Loss: {:.3f}".format(average_test_loss))
        print("Test Accuracy: {:.3f}%".format(accuracy*100))  
        
        #checkpoint save
        model.class_to_idx = class_to_idx
        checkpoint = {'input_size': args.hidden_units,
                   'output_size': 102,
                   'structure': args.arch,
                   'learning_rate': args.learning_rate,
                   'classifier': model.classifier,
                   'epochs': args.epochs,
                   'optimizer': optimizer.state_dict(),
                   'state_dict': model.state_dict(),
                   'class_to_idx': model.class_to_idx}

        torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))
        return True
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./")
    parser.add_argument('--arch', action="store", default="vgg16", choices=["vgg16", "alexnet"])
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,default=0.001)
    parser.add_argument('--gpu', dest='gpu', default = False)
    parser.add_argument('--epochs', dest='epochs',default=3, type=int)
    parser.add_argument('--hidden_units', dest = 'hidden_units',action="store",default = 2048)
    
    args = parser.parse_args()
    
    trainloader, validloader, testloader, class_to_idx = data_transform(args)
    model_training(args,trainloader, validloader, testloader, class_to_idx)
    
        