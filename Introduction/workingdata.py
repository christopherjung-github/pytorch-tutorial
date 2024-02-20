# Working with Data Section
import torch
from torch import nn
# Libraries below are two primitives in work with data
from torch.utils.data import DataLoader
from torchvision import datasets
# ToTensor library transforms the data from available datasets
from torchvision.transforms import ToTensor

download_variant = True

class trainingClass:
    def __init__(self):
        self.root = "data"
    
    def download(self):
        if self.root:
            root = self.root
        else:
            print("Class is not intiialized")
            return

        try:
            # Download the training data from open datasets
            ## Every TorchVision Dataset includes transform and target_transform to
            ## modify the samples and labels respectively
            self.training_data = datasets.FashionMNIST( # library contains Dataset objects
                root=root,
                train=True,
                download=True,
                transform=ToTensor()
            )

            # Download test data from open datasets
            self.test_data = datasets.FashionMNIST(
                root=root,
                train=False,
                download=True,
                transform=ToTensor()
            )
        except:
            print("Unable to download data from open datasets")
            return

    def getTrainingData(self):
        try:
            return self.training_data
        except:
            print("Training data was not downloaded from open datasets")
    
    def getTestData(self):
        try:
            return self.test_data
        except:
            print("Test data was not downloaded from open datasets")

trainingClass = trainingClass()

if trainingClass and download_variant == True:
    trainingClass.download()
    # print(trainingClass.training_data)
    # print(trainingClass.test_data)

batch_size = 64

# Get the datasets
training_data = trainingClass.getTrainingData()
test_data = trainingClass.getTestData()

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
print(train_dataloader)
print(test_dataloader)

for x, y in test_dataloader:
    print(f"Shape of x [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break