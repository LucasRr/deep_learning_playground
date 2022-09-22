import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_MNIST_data_loaders(batch_size=64):


	# Download train and test data:

	training_data_ = datasets.MNIST(
	    root="/Users/lucas/Documents/Data/",
	    train=True,
	    download=True,
	    transform=ToTensor(),
	)

	test_data = datasets.MNIST(
	    root="/Users/lucas/Documents/Data/",
	    train=False,
	    download=True,
	    transform=ToTensor(),
	)

	# split training data into train and validation sets:
	train_split, val_split = torch.utils.data.random_split(
		training_data_, [50000, 10000],
		generator=torch.Generator().manual_seed(42))

	# Create data loaders.
	train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_split, batch_size=batch_size)
	test_dataloader = DataLoader(test_data, batch_size=batch_size)

	return train_dataloader, val_dataloader, test_dataloader
