
import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms
from torchvision import utils

import torchvision as tv

from torchsummary import summary

import torch.optim


import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from utils import *

import pandas as pd

################################################################################
################################################################################
def saveDbOrder(file_name, lst):
    info    = {}
    nfiles  = 0

    for folder, name in lst:
        info[name]  = folder
        nfiles     += len(folder)

    u_saveDict2File(file_name, info )
    return nfiles 

################################################################################
################################################################################
def parseData():

    data_dir    = 'db/'
    data_lst    = 'lst/'

    u_mkdir(data_lst)

    tr, vl, ts = u_make_dataset_files(data_dir, '.jpg')
    
    nfiles = saveDbOrder(data_lst + 'train.json', tr)
    nfiles += saveDbOrder(data_lst + 'valid.json', vl)
    nfiles += saveDbOrder(data_lst + 'test.json', ts)

    print(nfiles)
    
################################################################################
################################################################################
def data2loader():
    data = u_loadJson('lst/train.json')
    
    #hist    = []
    #names   = []

    ##...........................................................................
    ## visualizing db data
    #for item in data:
    #    hist.append(len(data[item]))
    #    names.append(item)

    #values  = hist
    #ind = np.arange(len(names)) 
    #plt.bar(ind, values)
    #plt.xticks(ind, names, rotation='vertical')
    #plt.show()

    #..........................................................................
    # put db into loader

    transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    db_train = DbFromDict(data, transform= transformations)
    

################################################################################
################################################################################
################################################################################

class DbFromDict(Dataset):
    
    def __init__(self, info, transform=None):
        self.data_path, self.data_labels  = formatInput(info)
        self.transform  = transform        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        
        image = Image.open(data_path[index])
                
        if self.transform is not None:
            image = self.transform(image)
            
        return image, data_labels[index]
  
    # utils ....................................................................
    def formatInput(info):
        data_path   = []
        data_labels = []
        
        for lbl in info:
            for item in info[lbl]: 
                data_labels.append(lbl)
                data_path.apappend(item)

        return data_path, data_labels
          
################################################################################
################################################################################
if __name__ == '__main__':    
    data2loader()
   



    



