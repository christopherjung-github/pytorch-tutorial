# Creating Model Section
import torch
from torch import nn
# Libraries below are two primitives in work with data
from torch.utils.data import DataLoader
from torchvision import datasets
# ToTensor library transforms the data from available datasets
from torchvision.transforms import ToTensor

# Get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# Define model
## How to build a neural networks in PyTorch
## Inheritance of the Module class in the nn library
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Inherit the same 
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)