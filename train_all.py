import torch
from src.dataset import get_eurosat_dataloaders
from src.models import SimpleCNN, get_resnet18
from src.train import train_svm_hog, train_deep

DATA_DIR = "data"
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, classes = get_eurosat_dataloaders(DATA_DIR)

# SVM + HOG
train_svm_hog(train_loader.dataset, save_path=f"{MODELS_DIR}/svm_hog.pkl")

# Simple CNN
train_deep(SimpleCNN(len(classes)), train_loader,
           save_path=f"{MODELS_DIR}/simple_cnn.pth",
           device=DEVICE)

# ResNet18
resnet = get_resnet18(len(classes))
train_deep(resnet, train_loader,
           save_path=f"{MODELS_DIR}/resnet18.pth",
           device=DEVICE)
