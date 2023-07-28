import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import joblib
# Loaded the required libraries
 # Define your data directory and other variables
 data_dir = "C:\\Users\\Kshit\\Desktop\\tr"
 batch_size = 32
 random_seed = 42

 # Define transformations for the dataset
 transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

 # Load the dataset
 dataset = ImageFolder(data_dir, transform=transformations)