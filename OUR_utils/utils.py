import torch
import torch.nn as nn
import os 
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import fasttext
from Models.model import *   #It contains the Context Model

class ConTextDataset(Dataset):
    def __init__(self, directory_test_train_files, images_directory, fasttext, train = True, transform = False):
        
        self.path_test_train_files       = directory_test_train_files
        self.images_directory            = images_directory
        self.transform                   = transform

        if train:
            path_    = os.path.join(directory_test_train_files,'train.txt')         #Path to the training set 
            path_ocr = os.path.join(directory_test_train_files, 'ocr_train.txt') 
        else:
            path_ = os.path.join(directory_test_train_files,'test.txt')             #Path to the test set
            path_ocr = os.path.join(directory_test_train_files, 'ocr_test.txt')
        
        with open(path_, 'r') as file, open(path_ocr, 'r') as ocr_File:
            self.samples = [tuple(line.split()) for line in file]        #List of tuples.  Each tuple represents a file. A tuple contains the name and the label of the image.
            self.text    = [text.rstrip() for text in ocr_File]          #List of strings. Text of the ocr either for the train or test images.
            
        self.fasttext = fasttext
        self.dim_fasttext = self.fasttext.get_dimension()
        self.max_num_words = 64


    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_directory, self.samples[idx][0]+'.jpg')    #Image path
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)    #Returns a tensor after applying various transformations

        text = np.zeros((self.max_num_words, self.dim_fasttext))
        words = []
        if self.text[idx] != '0':
            for word in self.text[idx].split():
                if len(word) > 2: words.append(word)

        words = list(set(words))
        for i,w in enumerate(words):
            if i>=self.max_num_words: break
            text[i,:] = self.fasttext.get_word_vector(w)

        target = torch.tensor(int(self.samples[idx][1]))

        return image, text, target

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


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config, device="cuda"):
    # Make the data
    directory_test_train_files = '/home/xnmaster/data/'            
    images_directory           = '/home/xnmaster/data/JPEGImages/'

    fasttext_obj = fasttext.load_model('/home/xnmaster/cc.en.300.bin')
    train = ConTextDataset(directory_test_train_files, images_directory, fasttext_obj, train = True,  transform = get_transform(config.input_size, train = True))
    test  = ConTextDataset(directory_test_train_files, images_directory, fasttext_obj, train = False, transform = get_transform(config.input_size, train = False))

    train_loader  = make_loader(train, batch_size=config.batch_size)
    test_loader   = make_loader(test,  batch_size=config.batch_size)

    #Make the model
    #ConTextTransformer(num_classes = 28, dim = 256, depth = 2, heads = 4, mlp_dim = 512)
    model = ConTextTransformer(num_classes = config.classes, dim = config.dim, depth = config.depth, heads = config.heads, mlp_dim = config.mlp_dim, dropout=config.dropout)
    model = model.to(device)
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer