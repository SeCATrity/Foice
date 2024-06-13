import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision.io import read_image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
import pickle
import random
import os

class CVAE(nn.Module):
    def __init__(self, feature_size=256, latent_size=128):
        super(CVAE, self).__init__()
        
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512, out_features=256, bias=True)
        self.cnn.load_state_dict(torch.load("./F2V_models/CNN.pth"))
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()
        
        self.feature_size = feature_size

        # encode
        self.fc11  = nn.Sequential(
          nn.Linear(feature_size, 512),
          nn.ReLU(),
        )
        self.fc12  = nn.Sequential(
          nn.Linear(512, 256),
          nn.ReLU(),
        )
        self.fc13  = nn.Sequential(
          nn.Linear(256, latent_size),
          nn.ReLU(),
        )
        self.fc21 = nn.Linear(latent_size, latent_size)
        self.fc22 = nn.Linear(latent_size, latent_size)

        # decode
        self.fc3 = nn.Sequential(
          nn.Linear(latent_size+feature_size, 128),
          nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
        )
        self.fc5 = nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
        )
        self.fc6 = nn.Sequential(
          nn.Linear(256, feature_size),
          nn.ReLU(),
        )
        
    def encode(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        # print(torch.max(x[0]))
        h1 = self.fc11(x)
        # print(torch.max(h1[0]))
        h2 = self.fc12(h1)
        # print(torch.max(h2[0]))
        h3 = self.fc13(h2)
        # print(torch.max(h3[0]))
        z_mu = self.fc21(h3)
        z_var = self.fc22(h3)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        out = self.cnn(c)
        # print(torch.max(out[0]))
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-5)
        new_z = torch.cat((out, z), axis = 1)
        # print(f"new_z:{torch.max(new_z[0])}")
        # print(f'new_z:{torch.max(new_z[0])}')
        h3 = self.fc3(new_z)
        # print(f'h3:{torch.max(h3[0])}')
        h4 = self.fc4(h3)
        # print(f'h4:{torch.max(h4[0])}')
        h5 = self.fc5(h4)
        # print(f'h5:{torch.max(h5[0])}')
        h6 = self.fc6(h5)
        # print(f'h6:{torch.max(h6[0])}')
        embeds = h6
        # embeds = h4 / (torch.norm(h4, dim=1, keepdim=True) + 1e-5) 
        # print(f'embeds:{torch.max(embeds)}')
        return embeds, out

    def forward(self, label, image):
        mu, logvar = self.encode(label)
        z = self.reparameterize(mu, logvar)
        # print(f"min:{np.min(z.detach().cpu().numpy())}, max:{np.max(z.detach().cpu().numpy())}")
        return self.decode(z, image), mu, logvar