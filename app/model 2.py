<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os

# Load Model 1
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define model architecture (should match the training phase)
        self.fc = nn.Linear(256, 8)  # Example architecture

    def forward(self, x):
        return self.fc(x)

def load_model1():
    model1 = Model1()
    model1.load_state_dict(torch.load(os.path.join("Resnet50", "Resnet50.pt"), map_location=torch.device('cpu')))
    model1.eval()
    return model1

# Load Model 2
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Define model architecture (should match the training phase)
        self.fc = nn.Linear(256, 8)  # Example architecture

    def forward(self, x):
        return self.fc(x)

def load_model2():
    model2 = Model2()
    model2.load_state_dict(torch.load(os.path.join("yolov8", "yolov8.pt"), map_location=torch.device('cpu')))
    model2.eval()
    return model2

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image
=======
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os

# Load Model 1
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define model architecture (should match the training phase)
        self.fc = nn.Linear(256, 8)  # Example architecture

    def forward(self, x):
        return self.fc(x)

def load_model1():
    model1 = Model1()
    model1.load_state_dict(torch.load(os.path.join("Resnet50", "Resnet50.pt"), map_location=torch.device('cpu')))
    model1.eval()
    return model1

# Load Model 2
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Define model architecture (should match the training phase)
        self.fc = nn.Linear(256, 8)  # Example architecture

    def forward(self, x):
        return self.fc(x)

def load_model2():
    model2 = Model2()
    model2.load_state_dict(torch.load(os.path.join("yolov8", "yolov8.pt"), map_location=torch.device('cpu')))
    model2.eval()
    return model2

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image
>>>>>>> a75d4f660dc3168a3f772229327192caede06bb1
