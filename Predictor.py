import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder  # Added import for ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import joblib
from PIL import Image  # Added import for Image

# Load the trained model state dictionary
model_filename = "C:\\Users\\Kshit\\Desktop\\Waste Garbage Detection\\custom_resnet_model.joblib"
model_state_dict = joblib.load(model_filename)

# Define transformations for the input image
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Function to preprocess the input image and get the predictions
def predict_image(image_path, model_state_dict, class_names):
    image = Image.open(image_path)
    image = transformations(image).unsqueeze(0)

    # Load the model
    class CustomResNet(torch.nn.Module):
        def __init__(self, num_classes):
            super(CustomResNet, self).__init__()
            self.resnet = models.resnet18(pretrained=False)
            in_features = self.resnet.fc.in_features
            self.resnet.fc = torch.nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.resnet(x)

    num_classes = len(class_names)
    model = CustomResNet(num_classes)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)

    # Get the predicted class index
    _, preds = torch.max(outputs, 1)
    predicted_class_index = preds.item()

    # Map the predicted class index to the actual class label
    predicted_class_label = class_names[predicted_class_index]

    return predicted_class_label

# Load the dataset to get the class names
data_dir = "C:\\Users\\Kshit\\Desktop\\tr"
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformations)
class_names = dataset.classes

# Provide the path to the input image
input_image_path = "C:\\Users\\Kshit\\Desktop\\Waste Garbage Detection\\tester.jpg"

# Use the predict_image function to get the predictions
predicted_class_label = predict_image(input_image_path, model_state_dict, class_names)

print("Predicted Class:", predicted_class_label)
