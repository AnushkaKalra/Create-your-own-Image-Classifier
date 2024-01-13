import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import os
import random
import json
import argparse

def load_model(model_path):
    checkpoint = torch.load(model_path)
        
    model = torchvision.models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    print(image)
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image

def predict(image_path, model, topk, device, cat_to_name):
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # move to device
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    ps, top_cs = ps.topk(topk, dim=1)
    
    idx_to_flower = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_cs.tolist()[0]]

    return ps.tolist()[0], predicted_flowers_list

def display_predictions(args):
    if args.model_path is None:
        print("Error: Please provide the --model_path argument.")
        return

    if args.category_names is None:
        print("Error: Please provide the --category_names argument.")
        return

    model = load_model(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    top_ps, top_cs = predict(args.input, model, args.top_k, device, cat_to_name)
    
    print("Predictions are:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_cs[i], top_ps[i] * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
    parser.add_argument('--data_dir', action="store", dest="data_dir", default="./flowers/")
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest='category_names', action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store_true", dest="gpu")
    parser.add_argument('--model_path', dest='model_path')
    # parser.add_argument('--image_path', dest='image_path')
    
    args = parser.parse_args()
    
    display_predictions(args)
