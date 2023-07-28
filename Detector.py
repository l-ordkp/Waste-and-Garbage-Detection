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


 # Define the model architecture
 class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Create the model
 num_classes = len(dataset.classes)  # Number of output classes
 model = CustomResNet(num_classes)
 # Define the device to be used for computation (CPU)
 device = torch.device("cpu")
 model = model.to(device)

 # Define the optimization function and other hyperparameters
 learning_rate = 0.001
 optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 criterion = nn.CrossEntropyLoss()

 # Training loop
 def train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs=10):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_dl.dataset)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_loss = val_loss / len(val_dl.dataset)
        val_acc = corrects.double() / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history
 # Train the model
 num_epochs = 4
 history = train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs)
 # Save the trained model
 model_filename = "custom_resnet_model.joblib"
 joblib.dump(model.state_dict(), model_filename)
