import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt

import time

import utils
import utils.data
from utils.models import LogisticRegression, MLP_2layers, MLP_4layers
from utils.ML import train_model, evaluate_model, print_overall_metrics


batch_size = 1024
print(f"Using batch size = {batch_size}")

dataloaders = utils.data.get_MNIST_data_loaders(batch_size=batch_size)
train_dataloader, val_dataloader, test_dataloader = dataloaders

# device = 'cpu'
device = 'mps'

print(f"Using device {device}\n")


############## Logistic regression
print(" ============= ")

# Re-instantiate model to re-initialize weights:
model = LogisticRegression().to(device)

loss_fn = nn.CrossEntropyLoss()

# Define optimizer:
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Define number of epochs:
num_epochs = 10

# Train model:
train_loss_log, val_loss_log = train_model(
	model,
	train_dataloader,
	val_dataloader,
	optimizer,
	loss_fn,
	num_epochs,
	device=device)

print_overall_metrics(model, dataloaders, loss_fn, device=device)
print()

############## MLP 2 layers
print(" ============= ")

# Re-instantiate model to re-initialize weights:
model = MLP_2layers().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

train_loss_log, val_loss_log = train_model(
	model,
	train_dataloader,
	val_dataloader,
	optimizer,
	loss_fn,
	num_epochs,
	device=device)

print_overall_metrics(model, dataloaders, loss_fn, device=device)
print()


############## MLP 4 layers
print(" ============= ")

# Re-instantiate model to re-initialize weights:
model = MLP_4layers().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

train_loss_log, val_loss_log = train_model(
	model,
	train_dataloader,
	val_dataloader,
	optimizer,
	loss_fn,
	num_epochs,
	device=device)
print_overall_metrics(model, dataloaders, loss_fn, device=device)


