import os
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torchvision

def create_tuple(directory_test_train_files, images_directory, transform, resnet50, train = True):

    if train:
        path     = os.path.join(directory_test_train_files, 'train.txt')            #Path to the training set 
        path_ocr = os.path.join(directory_test_train_files, 'ocr_train.txt')
        pickle_file = '/home/xnmaster/data/train_data.pkl'
    else:
        path     = os.path.join(directory_test_train_files, 'test.txt')             #Path to the test set
        path_ocr = os.path.join(directory_test_train_files, 'ocr_test.txt')
        pickle_file = '/home/xnmaster/data/test_data.pkl'
    
    with open(path, 'r') as line:
        images_name_labels = line.readlines()

    with open(path_ocr, 'r') as ocr_File:
        ouput_ocr = [line.rstrip() for line in ocr_File]

    Data = []

    for image_name_label, ocr in zip(images_name_labels, ouput_ocr):
        image_name, label = image_name_label.split()
        image_path = os.path.join(images_directory, image_name + '.jpg')
        image      = Image.open(image_path).convert('RGB')
        
        image = transform(image)    #Returns a tensor after applying various transformations
        Data.append(resnet50(image), label, ocr)

    with open(pickle_file, 'wb') as file:
        pickle.dump(Data, file)

def get_transform(input_size, train = True):
    if train:
        data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return data_transforms_train
    
    else:
        data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return data_transforms_test

resnet50 = torchvision.models.resnet50(pretrained=True)     # Download the ResNet-50 architecture with pretrained weights.
modules  = list(resnet50.children())[:-2]                   # Keep all layers until the linear maps.
resnet50 = nn.Sequential(*modules)                          # Convert ResNet-50 modules to a sequential object.
for param in resnet50.parameters():                         # Freeze the weights to perform feature extraction.
    param.requires_grad = False

directory_test_train_files = '/home/xnmaster/data/'            
images_directory           = '/home/xnmaster/data/JPEGImages/'
create_tuple(directory_test_train_files, images_directory, get_transform(256, train = True) , resnet50, train = True)
create_tuple(directory_test_train_files, images_directory, get_transform(256, train = False), resnet50, train = False)