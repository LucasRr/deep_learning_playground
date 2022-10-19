import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt

import time

import utils
import utils.data, utils.ML, utils.models

torch.manual_seed(0)


device = 'mps'


batch_size = 512
print(f"Using batch size = {batch_size}")

dataloaders = utils.data.get_MNIST_data_loaders(batch_size=batch_size)
train_dataloader, val_dataloader, test_dataloader = dataloaders

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1, padding=1)   # 8x28x28
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=1)  # 16x28x28
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1) # 32x14x14
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1) # 64x7x7
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7,7), stride=1, padding=0) # 128x1x1
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128*1*1, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dense1(self.flatten(x))
        return x


model = SimpleCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()

num_epochs = 25

train_loss_log, val_loss_log = utils.ML.train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    device=device)