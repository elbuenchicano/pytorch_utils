
import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, models
from torchvision import transforms
from torchvision import utils

import torchvision as tv
from torch import optim, cuda

from torchsummary import summary
from timeit import default_timer as timer

import torch.optim


import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from utils import *
from utilsd_db import *

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
def data2loader(file, transformations= None, graph = True):
    data = u_loadJson(file)

    #...........................................................................
    # visualizing db data
    if (graph):
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

    return DbFromDict(data, transform= transformations)
    
################################################################################
################################################################################
def show(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0) )
    plt.imshow(npimg)
    plt.show()

################################################################################
################################################################################
def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

################################################################################
################################################################################
def trainModel():

    datadir = '/home/wjk68/'
    traindir = datadir + 'train/'
    validdir = datadir + 'valid/'
    testdir = datadir + 'test/'

    save_file_name = 'vgg16-transfer-4.pt'
    checkpoint_path = 'vgg16-transfer-4.pth'

    # Change to fit hardware
    batch_size = 4
    n_classes  = 100

    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    db_train    = data2loader('lst/train.json', image_transforms['train'], False)
    tr          = DataLoader(db_train, batch_size=batch_size, shuffle=True)

    db_train    = data2loader('lst/valid.json', image_transforms['train'], False)
    vl          = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    
    #imgs, steering_angle = next(iter(ld))
    #print('Batch shape:',imgs.numpy().shape)
    #show(utils.make_grid(imgs, 10))

    model = models.vgg16(pretrained=True)    
    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    model.to('cuda')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())      
    
    model, history = train(
    model,
    criterion,
    optimizer,
    tr,
    vl,
    save_file_name=save_file_name,
    max_epochs_stop=5,
    n_epochs=30,
    print_every=2)

################################################################################
################################################################################

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=5,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print('Model has been trained for: {} epochs.\n'.format(model.epochs))
    except:
        model.epochs = 0
        print('Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            #if train_on_gpu:
            data = data.cuda()
            target = target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print ( 'Epoch: {} \t, {} acc.'.format(epoch, accuracy))

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    #if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print('\nEpoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(epoch, train_loss,valid_loss))
                    #print('\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            '\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            '{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        '\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        '{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

################################################################################
################################################################################
if __name__ == '__main__':    
    trainModel()