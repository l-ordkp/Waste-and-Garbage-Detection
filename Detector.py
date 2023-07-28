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

# Split the dataset into training, validation, and test sets
 torch.manual_seed(random_seed)
 train_size = int(0.7 * len(dataset))
 val_size = int(0.1 * len(dataset))
 test_size = len(dataset) - train_size - val_size
 train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

 # Create data loaders for training, validation, and test sets
 train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
 val_dl = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=4, pin_memory=True)
 test_dl = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=4, pin_memory=True)