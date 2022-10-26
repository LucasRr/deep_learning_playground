'''

This script benchmarks various strategies to load
ImageNet data, from raw jpg or preprocessed tensors.

'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import json, os, glob, time

from utils.data import ImageNet, ImageNet_preprocessed

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
)])



if __name__ == "__main__":

	path_to_im_folder = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val/"
	path_to_preprocessed_folder = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val_preprocessed/"
	path_to_labels_txt = "/Users/lucas/Documents/Data/ImageNet/imagenet_2012_validation_label_idxs.txt"

	imagenet_data = ImageNet(path_to_im_folder, path_to_labels_txt, preprocess)
	imagenet_data_preprocessed = ImageNet_preprocessed(path_to_preprocessed_folder, path_to_labels_txt)

	batch_size = 32
	num_workers = 4

	print(f"batch_size: {batch_size}, num_workers: {num_workers}")

	dataloader = torch.utils.data.DataLoader(imagenet_data,
										batch_size=batch_size,
										num_workers=num_workers)

	dataloader_preprocessed = torch.utils.data.DataLoader(imagenet_data_preprocessed,
										batch_size=batch_size,
										num_workers=num_workers)


	num_images = len(imagenet_data)
	num_batches = num_images//batch_size


	# Image loading time:
	print("Load from jpeg:")
	t = time.time()
	for i, (im, l) in enumerate(imagenet_data):
		if i % 1000 == 0:
			print(f"\rloading image {i} / {num_images}", end="")
	dt = time.time()-t
	print(f"\nTotal time: {dt/60:.2f} min")


	# Image loading time:
	print("Load from preprocessed tensors:")
	t = time.time()
	for i, (im, l) in enumerate(imagenet_data_preprocessed):
		if i % 1000 == 0:
			print(f"\rloading image {i} / {num_images}", end="")
	dt = time.time()-t
	print(f"\nTotal time: {dt/60:.2f} min")


	# Batch loading time:
	t = time.time()
	for i, (X, y) in enumerate(dataloader_preprocessed):
		if i % 100 == 0:
			print(f"\rloading batch {i} / {num_batches}", end="")
	dt = time.time()-t
	print(f"\nTotal time: {dt/60:.2f} min")




