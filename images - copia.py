import time,os,json
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)



train_Set_dir = 'C:/Users/34644/Desktop/BUISNESS DETECTION/data/ImageSets/0/train.txt'
images_dir    = 'C:/Users/34644/Desktop/BUISNESS DETECTION/data/JPEGImages'


image_list = []
with open(train_Set_dir, 'r') as images:

    for line in images:

        line = line.strip()

        image_list.append(line)


img_name = os.path.join(self.root_dir, self.samples[idx][0]+'.jpg')
image = Image.open(img_name).convert('RGB')

train_Set_dir
print(image_list)

