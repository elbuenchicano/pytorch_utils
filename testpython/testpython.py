
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
def dataAugmentation():
    data = u_loadJson('lst/train.json')
    
    hist    = []
    names   = []

    for item in data:
        hist.append(len(data[item]))
        names.append(item)

    values  = hist

    ind = np.arange(len(names)) 
    plt.bar(ind, values)
    plt.xticks(ind, names, rotation='vertical')
    plt.show()

################################################################################
################################################################################
################################################################################

class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    


################################################################################
################################################################################
if __name__ == '__main__':    
    dataAugmentation()
    


    



