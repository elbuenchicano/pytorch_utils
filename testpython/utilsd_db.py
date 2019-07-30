import sys
import os
import argparse
import json
import re
import random 
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from utils import *
from PIL import Image
import matplotlib.pyplot as plt

################################################################################
################################################################################
def ud_make_dataset_files (directory, token, split= (.5, .25, .25), shuffle= True):

    files       = u_listFileAll_(directory, token)
    n_clases    = len(files)
    split       = np.array(split)

    train = []
    valid = [] 
    test  = []

    #...........................................................................
    for dir, lfile in files:
        if shuffle:
            random.shuffle(lfile)

        bounds  = ( split * len(lfile) ).astype(int)
        
        train.append( [ lfile[:bounds[0]], 
                        os.path.basename(dir) ] )
        valid.append( [ lfile[bounds[0]: bounds[0] + bounds[1]], 
                        os.path.basename(dir) ] )
        test.append( [ lfile[bounds[0] + bounds[1]:], 
                       os.path.basename(dir) ] )

    return train, valid, test

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class DbFromDict(Dataset):
    
    def __init__(self, info, transform=None):
        self.data_path, self.data_labels  = self.formatInput(info)
        self.transform  = transform  
        self.src        = info

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)


        image = Image.open(self.data_path[index])
        
        if len(image.getbands()) == 1:
            image = np.stack((image,)*3, axis=-1)
            image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, self.data_labels[index]
  
    # utils ....................................................................
    def formatInput(self, info):
        data_path   = []
        data_labels = []
        
        cls = 0
        for lbl in info:
            for item in info[lbl]: 
                data_labels.append(cls)
                data_path.append(item)

            cls += 1

        return data_path, torch.tensor(data_labels)

