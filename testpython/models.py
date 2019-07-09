import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


###################################################################################################
###################################################################################################
###################################################################################################
#### VARIATIONAL AUTOENCODER for 28 * 28 * 1 images

class VAE(nn.Module):
    def __init__(self, latent_variable_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2m = nn.Linear(400, latent_variable_dim) # use for mean
        self.fc2s = nn.Linear(400, latent_variable_dim) # use for standard deviation
        
        self.fc3 = nn.Linear(latent_variable_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def reparameterize(self, log_var, mu):
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a iid standard normal same shape as s
        return eps.mul(s).add_(mu)
        
    def forward(self, input):
        x = input.view(-1, 784)
        x = torch.relu(self.fc1(x))
        log_s = self.fc2s(x)
        m = self.fc2m(x)
        z = self.reparameterize(log_s, m)
        
        x = self.decode(z)
        
        return x, m, log_s
    
    def decode(self, z):
        x = torch.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))
        return x
###################################################################################################
###################################################################################################

def lossVAE(input_image, recon_image, mu, log_var):
    CE = F.binary_cross_entropy(recon_image, input_image.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD + CE

###################################################################################################
###################################################################################################
###################################################################################################
#### CONVOLUTIONAL AUTOENCODER for 32 * 32 * 1 images


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)

        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.c4 = nn.Conv2d(32, 32, 3, padding=1)
        

        self.mp = nn.MaxPool2d((2,2), stride= 2)

        self.up = Interpolate(scale_factor= 2, mode='nearest')

        self.cf = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input):
        x = self.mp(torch.relu(self.c1(input)))
        x = self.mp(torch.relu(self.c2(x)))
        x = self.up(torch.relu(self.c3(x)))
        x = self.up(torch.relu(self.c4(x)))
        x = torch.sigmoid(self.cf(x))
     
        return x


def lossCAE(input_image, recon_image,):
    CE = F.binary_cross_entropy(recon_image, input_image, reduction='sum')
    return CE

###################################################################################################
###################################################################################################
###################################################################################################
#### VANILLA VGG16 ARCH for 224x224x3 images

class VGG11(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self.c1 = nn.Conv2d(1, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.f1 = nn.Linear(64)
        self.f1 = nn.Linear(64)
        self.f1 = nn.Linear(32)



        self.mp = nn.MaxPool2d((2,2), stride= 2)

